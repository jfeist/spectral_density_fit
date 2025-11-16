import numpy as np
import nlopt
import warnings
import jax
import jax.numpy as jnp
from jax import grad, jacobian, jit

from .spectral_densities import Jmod_naive, _non_jitted_Jmod


class spectral_density_fitter(nlopt.opt):
    """Fit arbitrary spectral densities with a few-mode model.

    This class implements the fitting procedure from I. Medina, F. J. García-Vidal,
    A. I. Fernández-Domínguez, and J. Feist, Phys. Rev. Lett. 126, 093601 (2021),
    https://doi.org/10.1103/PhysRevLett.126.093601, to represent a spectral density
    J(ω) by a few-mode model with effective Hamiltonian H_eff and coupling g.

    The fitter optimizes the parameters of the effective Hamiltonian and coupling
    to minimize the difference between the target spectral density and the model
    spectral density computed from the few-mode parameters.

    Parameters
    ----------
    ω : array-like
        A 1D array of frequencies at which the spectral density is provided.
    J : array-like
        The target spectral density. Can be:
        
        - 1D array of shape (Nω,) for a single emitter
        - 3D array of shape (Ne, Ne, Nω) for multiple emitters
        
    Hgtmpl : int or tuple
        Template for the Hamiltonian and coupling. Can be:
        
        - An integer Nm specifying the number of modes (all elements allowed to vary)
        - A tuple (Htmpl, gtmpl) of boolean arrays specifying which elements can be nonzero:
        
          - Htmpl: shape (Nm, Nm), template for effective Hamiltonian
          - gtmpl: shape (Ne, Nm), template for coupling matrix
    λlims : tuple or None, optional
        Limits for the eigenvalues of the Hamiltonian. If None (default), uses
        (ω.min(), ω.max()). Set to False to disable eigenvalue constraints.
    fitlog : bool, optional
        If True, minimize the error in log-space instead of linear space.
        Only supported for single emitter (Ne=1). Default is False.
    diagonalize : bool or None, optional
        Whether to use diagonalization method (True) or direct inversion (False).
        If None (default), automatically chooses based on device: True for CPU,
        False for GPU.
    device : jax.Device or None, optional
        JAX device to use for computations. If None (default), chooses based on
        diagonalize parameter.
    algorithm : nlopt algorithm, optional
        NLopt optimization algorithm to use. Default is nlopt.LD_CCSAQ
        (gradient-based CCSAQ algorithm).

    Attributes
    ----------
    ω : jax.numpy.ndarray
        Frequency array
    J : jax.numpy.ndarray
        Target spectral density
    Ne : int
        Number of emitters
    Nm : int
        Number of modes
    Htmpl : jax.numpy.ndarray
        Template for Hamiltonian
    gtmpl : jax.numpy.ndarray
        Template for coupling
    Nps : int
        Number of fit parameters
    Hκg_to_ps : callable
        Function to convert (H, κ, g) to parameter vector
    ps_to_Hκg : callable
        Function to convert parameter vector to (H, κ, g)
    Jfun : callable
        Function to compute spectral density from parameters
    obj_fun : callable
        Objective function for optimization

    Examples
    --------
    >>> import jax
    >>> import numpy as np
    >>> from spectral_density_fit import spectral_density_fitter
    >>> 
    >>> # Enable 64-bit precision for better accuracy
    >>> jax.config.update("jax_enable_x64", True)
    >>> 
    >>> # Create frequency array and target spectral density
    >>> ω = np.linspace(-3, 3, 100)
    >>> J_target = 0.1 / (ω**2 + 0.1**2)  # Lorentzian spectral density
    >>> 
    >>> # Fit with 3 modes
    >>> fitter = spectral_density_fitter(ω, J_target, 3)
    >>> 
    >>> # Initial guess (random)
    >>> ps0 = np.random.normal(size=fitter.Nps)
    >>> 
    >>> # Optimize
    >>> ps_opt = fitter.optimize(ps0)
    >>> 
    >>> # Get fitted spectral density
    >>> J_fit = fitter.Jfun(ω, ps_opt)
    """
    def __init__(self, ω, J, Hgtmpl, λlims=None, fitlog=False, diagonalize=None, device=None, algorithm=nlopt.LD_CCSAQ):
        if not jax.config.jax_enable_x64:
            warnings.warn(
                """jax is not using 64bit precision, this can affect the fitting accuracy.
            call jax.config.update("jax_enable_x64", True) at the start of the script to change this.""",
                RuntimeWarning,
            )

        if diagonalize is None:
            # if diagonalize is not set explicitly, default to it while running on CPU
            # but use direct inversion on GPU (since non-Hermitian diagonalization is
            # not available on GPUs, but they are so much faster that the "worse"
            # algorithm ends up being better)
            diagonalize = jax.default_backend() == "cpu"

        if device is None:
            device = jax.devices("cpu")[0] if diagonalize else jax.devices()[0]

        self.device = device
        self.diagonalize = diagonalize
        self.fitlog = fitlog
        self.λlims = λlims
        self.algorithm = algorithm

        with jax.default_device(self.device):
            self.ω = jnp.array(ω)
            J = jnp.array(J)
            if J.ndim == 1:
                J = J[None, None, :]
            self.J = J

            if isinstance(Hgtmpl, tuple):
                self.Htmpl, self.gtmpl = Hgtmpl
                self.Ne, self.Nm = self.gtmpl.shape
                if self.Htmpl.shape != (self.Nm, self.Nm):
                    raise ValueError(f"Shapes for Htmpl ({self.Htmpl.shape}) and gtmpl ({self.gtmpl.shape}) are not consistent. Should be (Nm,Nm) and (Ne,Nm).")
            else:
                self.Nm = int(Hgtmpl)
                self.Ne = J.shape[0]
                self.Htmpl = jnp.ones((self.Nm, self.Nm))
                self.gtmpl = jnp.ones((self.Ne, self.Nm))

        if self.J.shape != (self.Ne, self.Ne, len(self.ω)):
            raise ValueError(f"Input shapes are not consistent. Should be (Ne,Ne,Nω) for J (got {self.J.shape}), (Nω,) for ω (got {self.ω.shape}), and (Ne,Nm) for gtmpl (got {self.gtmpl.shape}).")

        if self.fitlog and self.Ne > 1:
            raise ValueError(f"fitlog=True only supported for 1 emitter. Got Ne = {self.Ne}.")

        Jmodfun = _non_jitted_Jmod if self.diagonalize else Jmod_naive
        Nps, Hκg_to_ps, ps_to_Hκg, Jfun, obj_fun = make_jax_closures(self.ω, self.J, self.Htmpl, self.gtmpl, self.fitlog, Jmodfun, self.device)
        self.Nps = Nps
        self.Hκg_to_ps = Hκg_to_ps
        self.ps_to_Hκg = ps_to_Hκg
        self.Jfun = Jfun
        self.obj_fun = obj_fun

        super().__init__(algorithm, Nps)
        self.set_min_objective(obj_fun)
        self.set_ftol_rel(1e-5)

        if λlims is not False:
            λmin, λmax = (ω.min(), ω.max()) if λlims is None else λlims
            nlopt_constraints = make_jax_constraints(λmin, λmax, ps_to_Hκg)
            self.add_inequality_mconstraint(nlopt_constraints, np.zeros(2 * self.Nm))


def make_jax_closures(ω, J, Htmpl, gtmpl, fitlog, Jmodfun, device):
    """Create JAX-compiled closures for the fitting procedure.

    This function creates the necessary functions for converting between
    the parameter space used by the optimizer and the physical parameters
    (Hamiltonian, decay rates, coupling), as well as the objective function.

    Parameters
    ----------
    ω : jax.numpy.ndarray
        Frequency array
    J : jax.numpy.ndarray
        Target spectral density
    Htmpl : jax.numpy.ndarray
        Template for Hamiltonian (indicates which elements can be nonzero)
    gtmpl : jax.numpy.ndarray
        Template for coupling (indicates which elements can be nonzero)
    fitlog : bool
        If True, minimize error in log-space
    Jmodfun : callable
        Function to compute spectral density (Jmod or Jmod_naive)
    device : jax.Device
        JAX device to use for computations

    Returns
    -------
    Nps : int
        Number of fit parameters
    Hκg_to_ps : callable
        Function to convert (H, κ, g) to parameter vector
    ps_to_Hκg : callable
        Function to convert parameter vector to (H, κ, g)
    Jfun : callable
        Function to compute spectral density from parameters
    obj_fun : callable
        Objective function for NLopt optimization (returns error and gradient)
    """
    with jax.default_device(device):
        Ne, Nm = gtmpl.shape

        # get the indices of the nonzero entries in the upper triangle of Htmpl
        H_inds = np.nonzero(np.triu(Htmpl))
        # get the indices of the nonzero entries in gtmpl
        g_inds = np.nonzero(gtmpl)
        Nps_H = len(H_inds[0])
        Nps_g = len(g_inds[0])
        if jnp.iscomplexobj(J):
            # for complex J and thus g, we need 2 Nps_g real fit parameters
            # to represent the real and imaginary parts of g
            Nps_g *= 2

        Nps = Nps_g + Nm + Nps_H

        tmpH = jnp.zeros((Nm, Nm))
        tmpg = jnp.zeros((Ne, Nm), dtype=J.dtype)

    def Hκg_to_ps(H, κ, g):
        with jax.default_device(device):
            # the astype ensures that g is complex if J is complex, and then .view gives a real array of twice the size
            ps = jnp.hstack((g[g_inds].astype(J.dtype).view(κ.dtype), jnp.sqrt(κ), H[H_inds]))
        assert ps.shape == (Nps,)
        # commit to the device
        return jax.device_put(ps, device)

    @jit
    def ps_to_Hκg(ps):
        with jax.default_device(device):
            gps, sqrtκ, Hps = jnp.split(ps, [Nps_g, Nps_g + Nm])
            κ = sqrtκ**2
            # the .view ensures that the real array gps is viewed as a complex one if J is complex
            g = tmpg.at[g_inds].set(gps.view(J.dtype))
            Hu = tmpH.at[H_inds].set(Hps)
            H = Hu + jnp.tril(Hu.T, -1)
        return H, κ, g

    def _Jfun(ω, ps):
        H, κ, g = ps_to_Hκg(ps)
        Heff = H - 0.5j * jnp.diag(κ)
        return Jmodfun(ω, Heff, g)

    Jfun = jit(_Jfun)

    @jit
    def err(ps):
        Jf = _Jfun(ω, ps)
        assert jnp.iscomplexobj(Jf) == jnp.iscomplexobj(J)
        if fitlog:
            return jnp.linalg.norm(jnp.log(Jf) - jnp.log(J))
        else:
            return jnp.linalg.norm(Jf - J)

    grad_err = jit(grad(err))

    def nlopt_f(ps, grad=None):
        if grad is not None and grad.size > 0:
            grad[...] = grad_err(ps)
        return float(err(ps))

    return Nps, Hκg_to_ps, ps_to_Hκg, Jfun, nlopt_f


def make_jax_constraints(λmin, λmax, ps_to_Hκg):
    """Create constraint functions for NLopt to keep eigenvalues within bounds.

    Parameters
    ----------
    λmin : float
        Minimum allowed eigenvalue
    λmax : float
        Maximum allowed eigenvalue
    ps_to_Hκg : callable
        Function to convert parameter vector to (H, κ, g)

    Returns
    -------
    nlopt_constraints : callable
        Constraint function for NLopt that ensures eigenvalues of H are
        within [λmin, λmax]
    """
    @jit
    def f_constraints(ps):
        "constraint function that forces eigenvalues to be within the range [λmin,λmax]"
        H, κ, g = ps_to_Hκg(ps)
        λs = jnp.linalg.eigvalsh(H)
        # nlopt enforces constraint functions to be smaller than 0
        return jnp.hstack((λs - λmax, λmin - λs))

    jac_constraints = jit(jacobian(f_constraints))

    def nlopt_constraints(result, ps, grad):
        result[...] = f_constraints(ps)
        if grad.size > 0:
            grad[...] = jac_constraints(ps)

    return nlopt_constraints
