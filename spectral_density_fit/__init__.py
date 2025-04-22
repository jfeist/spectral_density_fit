"""Fit arbitrary spectral densities with a few-mode model

Implements the procedure from I. Medina, F. J. García-Vidal, A. I.
Fernández-Domínguez, and J. Feist, Phys. Rev. Lett. 126, 093601 (2021),
https://doi.org/10.1103/PhysRevLett.126.093601
"""

__version__ = "0.2.0"
__all__ = ["spectral_density_fitter"]

import jax
import jax.numpy as jnp
from jax import grad, jacobian, jit
from functools import partial

import numpy as np
import nlopt
import warnings


@jit
def Jmod_lorentz(ω, λs, g):
    χ = jnp.einsum("il,jl,lw->ijw", g, g, 1 / (λs[:, None] - ω[None, :]))
    return χ.imag / jnp.pi


@jit
def Jmod_naive(ω, Heff, g):
    """Compute the spectral density matrix J(ω) using a naive method that solves
    the linear equation at each frequency. This method avoids diagonalizing the
    effective Hamiltonian `Heff`, making it compatible with accelerators like
    GPUs where diagonalization of non-Hermitian matrices is not implemented. It
    is less efficient than diagonalization but allows for direct gradient
    computation with JAX, making it useful for verification and optimization
    tasks.

    Parameters:
    -----------
    ω : array-like
        A 1D array of frequencies at which to compute the spectral density.
    Heff : array-like
        The effective Hamiltonian matrix, assumed to be of shape (N, N).
    g : array-like
        The coupling vector or matrix, of shape (N,) or (N, M), where N is the
        dimension of the Hamiltonian and M is the number of coupling channels.

    Returns:
    --------
    array-like
        The spectral density matrix J(ω), with shape (M, M, len(ω)).

    Notes:
    ------
    - If `g` is complex and has more than one element, the resulting spectral
      density matrix J(ω) is Hermitian with complex off-diagonal elements. In
      this case, Heff does not need to be complex symmetric.
    - If `g` is real or has only one element, `Heff` is assumed to be complex
      symmetric, and the resulting spectral density matrix J(ω) is
      explicitly real.
    """
    II = jnp.eye(Heff.shape[0])
    Hω = Heff[None, :, :] - ω[:, None, None] * II[None, :, :]
    # χ = 1/(H-w)
    # χ = χ^T -> χ^* = χ^\dagger
    # Im(χ) = (χ-χ^*) / 2i = (χ-χ^\dagger) / 2i
    # J = g @ Im(χ) @ g^\dagger / π = g @ (χ-χ^\dagger) @ g^\dagger / 2iπ
    # R = g @ χ @ g^\dagger
    # J = (R - R^\dagger) / 2iπ
    R = g @ jnp.linalg.solve(Hω, g.conj().T[None, :, :])
    if jnp.iscomplexobj(g) and g.shape[0] > 1:
        # if g is complex and Ne>1, J is a Hermitian matrix with complex off-diagonal elements.
        # indices of R are (w,i,j), R^\dagger = R.conj()[w,j,i], transpose to (i,j,w)
        return (R.transpose(1, 2, 0) - R.conj().transpose(2, 1, 0)) / (2j * jnp.pi)
    else:
        # we assume that H is complex symmetric, so for real g, R = R^T
        # -> J = R.imag / π (and is thus explicitly real)
        return R.imag.transpose(1, 2, 0) / jnp.pi


# fχ gives a jax tracer leak error if jitted in a function called with ω as an
# explicit argument in newer jax versions (certainly in 0.4.14, not sure since
# when, 0.2.19 was fine) due to the custom_jvp definition. So define the function
# without jit, and below provide the jitted wrapper
def _non_jitted_Jmod(ω, Heff, g):
    # χ = 1/(H-w)
    χ = fχ(ω, Heff)
    # J = g Im(χ) g^\dagger / π
    # the .astype(g.dtype) ensures that no warnings are raised
    return jnp.einsum("il,lmw,jm->ijw", g, χ.imag.astype(g.dtype), g.conj()) / jnp.pi


def Jmod(ω, Heff, g):
    """Compute the spectral density matrix J(ω) by diagonalizing the effective
    Hamiltonian `Heff`, which is assumed to be complex symmetric. This method is
    in principle more efficient than solving the linear equation at each
    frequency, but for now is not compatible with accelerators like GPUs, for
    which diagonalization of non-Hermitian matrices is not implemented in jax.
    The gradient of complex matrix diagonalization is not provided by jax, so
    the gradient of the crucial step in the algorithm is implemented manually.

    Parameters:
    -----------
    ω : array-like
        A 1D array of frequencies at which to compute the spectral density.
    Heff : array-like
        The effective Hamiltonian matrix, assumed to be of shape (N, N), and to
        be complex symmetric.
    g : array-like
        The coupling vector or matrix, of shape (N,) or (N, M), where N is the
        dimension of the Hamiltonian and M is the number of coupling channels.

    Returns:
    --------
    array-like
        The spectral density matrix J(ω), with shape (M, M, len(ω)). It is real if
        `g` is real, and complex if `g` is complex.
    """
    # diagonalization is not supported on GPUs, so we need to run this on CPU
    with jax.default_device(jax.devices("cpu")[0]):
        return jit(_non_jitted_Jmod)(ω, Heff, g)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def fχ(ω, Heff):
    λs, V = jnp.linalg.eig(Heff)
    V /= jnp.sqrt(jnp.sum(V**2, axis=0))
    return jnp.einsum("il,lw,jl->ijw", V, 1 / (λs[:, None] - ω[None, :]), V)


@fχ.defjvp
def fχ_jvp(ω, primals, tangents):
    # χ = 1/(H-w)
    # express dχ in terms of χ, H, dH by using that it solves the linear equation
    # (H-w) χ = 1
    # ∂H χ + (H-w) ∂χ = 0
    # ∂χ = 1/(H-w) (-∂H χ) = -χ ∂H χ
    (Heff,) = primals
    (dHeff,) = tangents

    λs, V = jnp.linalg.eig(Heff)
    V /= jnp.sqrt(jnp.sum(V**2, axis=0))
    λωinv = 1 / (λs[:, None] - ω[None, :])

    χ = jnp.einsum("il,lw,jl->ijw", V, λωinv, V)
    dχ = jnp.einsum("imw,ml,ljw->ijw", χ, -dHeff, χ)
    return χ, dχ


# version that allows to pass "templates" for H and g that
# indicate where they are allowed to be nonzero in the fit
def spectral_density_fitter(ω, J, Hgtmpl, λlims=None, fitlog=False, diagonalize=None, device=None, algorithm=nlopt.LD_CCSAQ):
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

    with jax.default_device(device):
        ω = jnp.array(ω)
        J = jnp.array(J)
        if J.ndim == 1:
            J = J[None, None, :]

        if isinstance(Hgtmpl, tuple):
            Htmpl, gtmpl = Hgtmpl
            Ne, Nm = gtmpl.shape
            assert Htmpl.shape == (Nm, Nm)
        else:
            Nm = int(Hgtmpl)
            Ne = J.shape[0]
            Htmpl = jnp.ones((Nm, Nm))
            gtmpl = jnp.ones((Ne, Nm))

        assert J.shape == (Ne, Ne, len(ω))

        if fitlog and Ne > 1:
            raise ValueError(f"fitlog=True only supported for 1 emitter. Got Ne = {Ne}.")

        # get the indices of the nonzero entries in the upper triangle of Htmpl
        H_inds = np.nonzero(np.triu(Htmpl))
        # get the indices of the nonzero entries in gtmpl
        g_inds = np.nonzero(gtmpl)
        Nps_H = len(H_inds[0])
        Nps_g = len(g_inds[0])

        tmpH = jnp.zeros((Nm, Nm))
        tmpg = jnp.zeros((Ne, Nm))

        if jnp.iscomplexobj(J):
            Nps = 2 * Nps_g + Nm + Nps_H

            def Hκg_to_ps(H, κ, g):
                with jax.default_device(device):
                    ps = jnp.hstack((g[g_inds].real, g[g_inds].imag, jnp.sqrt(κ), H[H_inds]))
                # commit to the device
                return jax.device_put(ps, device)

            @jit
            def ps_to_Hκg(ps):
                gps, gpsc, sqrtκ, Hps = jnp.split(ps, [Nps_g, 2 * Nps_g, 2 * Nps_g + Nm])
                κ = sqrtκ**2
                g = tmpg.at[g_inds].set(gps) + 1j * tmpg.at[g_inds].set(gpsc)
                Hu = tmpH.at[H_inds].set(Hps)
                H = Hu + jnp.tril(Hu.T, -1)
                return H, κ, g
        else:
            Nps = Nps_g + Nm + Nps_H

            def Hκg_to_ps(H, κ, g):
                with jax.default_device(device):
                    ps = jnp.hstack((g[g_inds], jnp.sqrt(κ), H[H_inds]))
                # commit to the device
                return jax.device_put(ps, device)

            @jit
            def ps_to_Hκg(ps):
                gps, sqrtκ, Hps = jnp.split(ps, [Nps_g, Nps_g + Nm])
                κ = sqrtκ**2
                g = tmpg.at[g_inds].set(gps)
                Hu = tmpH.at[H_inds].set(Hps)
                H = Hu + jnp.tril(Hu.T, -1)
                return H, κ, g

        if diagonalize:

            def Jfun(ω, ps):
                H, κ, g = ps_to_Hκg(ps)
                Heff = H - 0.5j * jnp.diag(κ)
                JJ = _non_jitted_Jmod(ω, Heff, g)
                return JJ
        else:

            def Jfun(ω, ps):
                H, κ, g = ps_to_Hκg(ps)
                Heff = H - 0.5j * jnp.diag(κ)
                JJ = Jmod_naive(ω, Heff, g)
                return JJ

        @jit
        def err(ps):
            Jf = Jfun(ω, ps)
            assert jnp.iscomplexobj(Jf) == jnp.iscomplexobj(J)
            if fitlog:
                return jnp.linalg.norm(jnp.log(Jf) - jnp.log(J))
            else:
                return jnp.linalg.norm(Jf - J)

        grad_err = jit(grad(err))

        def nlopt_f(ps, grad):
            if grad.size > 0:
                grad[...] = grad_err(ps)
            return float(err(ps))

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

        opt = nlopt.opt(algorithm, Nps)
        opt.set_min_objective(nlopt_f)
        opt.set_ftol_rel(1e-5)

        if λlims is not False:
            λmin, λmax = (ω[0], ω[-1]) if λlims is None else λlims
            opt.add_inequality_mconstraint(nlopt_constraints, np.zeros(2 * Nm))

        # add members to opt to have everything in a single object
        opt.Hκg_to_ps = Hκg_to_ps
        opt.ps_to_Hκg = ps_to_Hκg
        opt.Jfun = jit(Jfun)  # we can jit this because it is not used in the optimization
        opt.obj_fun = nlopt_f

    return opt
