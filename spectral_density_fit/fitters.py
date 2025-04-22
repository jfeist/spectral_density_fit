import numpy as np
import nlopt
import warnings
import jax
import jax.numpy as jnp
from jax import grad, jacobian, jit

from .spectral_densities import Jmod_naive, _non_jitted_Jmod


class spectral_density_fitter(nlopt.nlopt.opt):
    # version that allows to pass "templates" for H and g that
    # indicate where they are allowed to be nonzero in the fit
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
            if jnp.iscomplexobj(J):
                Nps_g *= 2
            Nps = Nps_g + Nm + Nps_H

            tmpH = jnp.zeros((Nm, Nm))
            tmpg = jnp.zeros((Ne, Nm),dtype=J.dtype)

            def Hκg_to_ps(H, κ, g):
                with jax.default_device(device):
                    # the astype ensures that g is complex if J is complex, and then .view gives a real array of twice the size
                    ps = jnp.hstack((g[g_inds].astype(J.dtype).view(κ.dtype), jnp.sqrt(κ), H[H_inds]))
                assert ps.shape == (Nps,)
                # commit to the device
                return jax.device_put(ps, device)

            @jit
            def ps_to_Hκg(ps):
                gps, sqrtκ, Hps = jnp.split(ps, [Nps_g, Nps_g + Nm])
                κ = sqrtκ**2
                # the .view ensures that the real array gps is viewed as a complex one if J is complex
                g = tmpg.at[g_inds].set(gps.view(J.dtype))
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

            super().__init__(algorithm, Nps)
            self.set_min_objective(nlopt_f)
            self.set_ftol_rel(1e-5)

            if λlims is not False:
                λmin, λmax = (ω[0], ω[-1]) if λlims is None else λlims
                self.add_inequality_mconstraint(nlopt_constraints, np.zeros(2 * Nm))

            # add members to opt to have everything in a single object
            self.Hκg_to_ps = Hκg_to_ps
            self.ps_to_Hκg = ps_to_Hκg
            self.Jfun = jit(Jfun)  # we can jit this because it is not used in the optimization
            self.obj_fun = nlopt_f
