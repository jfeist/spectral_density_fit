"""Fit arbitrary spectral densities with a few-mode model

Implements the procedure from I. Medina, F. J. García-Vidal, A. I.
Fernández-Domínguez, and J. Feist, Phys. Rev. Lett. 126, 093601 (2021),
https://doi.org/10.1103/PhysRevLett.126.093601
"""

__version__ = '0.1.0'

from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import jax
import jax.numpy as jnp
from jax import grad, jacobian, jit
from functools import partial

import numpy as np
import nlopt

@jit
def Jmod_lorentz(ω,λs,g):
    χ = jnp.einsum('il,jl,wl->wij',g,g,1/(λs[None,:]-ω[:,None]))
    return χ.imag/jnp.pi

"""this actually solves the linear equation at each frequency. it is much less
efficient than diagonalizing Heff, but jax can calculate the gradient directly,
so we can use it as a check for our implementation.
"""
@jit
def Jmod_naive(ω,Heff,g):
    I = jnp.eye(Heff.shape[0])
    Hω = Heff[None,:,:] - ω[:,None,None]*I[None,:,:]
    A = jnp.linalg.solve(Hω,g.T[None,:,:])
    χ = g @ A
    return χ.imag/jnp.pi

@partial(jax.custom_jvp, nondiff_argnums=(0,))
def fχ(ω,Heff,g):
    λs, V = jnp.linalg.eig(Heff)
    V /= jnp.sqrt(jnp.sum(V**2,axis=0))
    
    G = g @ V
    return jnp.einsum('il,wl,jl->wij',G,1/(λs[None,:]-ω[:,None]),G)

def fχ_jvp(ω,primals,tangents):
    # χ = g 1/(H-w) g^T
    # H = V @ diag(λ) @ V^T
    # G = g V
    # ∂G = ∂g V

    # express dx in terms of x, g, H dH, dg by using that it solves the linear equation
    # x = 1/(H-w) g^T
    # (H-w) x = g^T
    # ∂H x + (H-w) ∂x = ∂g^T
    # ∂x = 1/(H-w) (∂g^T - ∂H x)
    
    # χ = g x
    # ∂χ = ∂g x + g ∂x
    # ∂g x = ∂g 1/(H-w) g^T = ∂G 1/(Λ-w) G^T
    
    # g ∂x = g 1/(H-w) (∂g^T - ∂H x)
    # g ∂x = G 1/(Λ-w) (∂G^T - V^T ∂H V 1/(Λ-w) G^T)
    
    Heff, g = primals
    dHeff, dg = tangents
    
    λs, V = jnp.linalg.eig(Heff)
    V /= jnp.sqrt(jnp.sum(V**2,axis=0))
    G = g @ V
    dG = dg @ V
    λωinv = 1/(λs[None,:]-ω[:,None])
    
    χ = jnp.einsum('il,wl,jl->wij',G,λωinv,G)

    # ∂g x = ∂G 1/(Λ-w) G^T
    dgx = jnp.einsum('il,wl,jl->wij',dG,λωinv,G)

    # g ∂x = G 1/(Λ-w) (∂G^T - V^T ∂H V 1/(Λ-w) G^T)
    # g dx = G 1/(Λ-w) VTdx_b
    VTdx_b = dG.T[None,:,:] - jnp.einsum('il,wl,jl->wij',V.T@dHeff@V,λωinv,G)
    gdx = jnp.einsum('il,wl,wlj->wij',G,λωinv,VTdx_b)
    
    dχ = dgx + gdx
    
    return χ, dχ

fχ.defjvp(fχ_jvp)

@jit
def Jmod(ω,Heff,g):
    χ = fχ(ω,Heff,g)
    return χ.imag/jnp.pi

# version that allows to pass "templates" for H and g that
# indicate where they are allowed to be nonzero in the fit
def make_objfun_shaped(ω,J,Htmpl,gtmpl,λlims=None,fitlog=False):
    Ne, Nm = gtmpl.shape
    assert Htmpl.shape == (Nm,Nm)

    J = jnp.array(J)
    assert J.shape == (len(ω),Ne,Ne)
    
    # get the indices of the nonzero entries in the upper triangle of Htmpl
    H_inds = np.nonzero(np.triu(Htmpl))
    # get the indices of the nonzero entries in gtmpl
    g_inds = np.nonzero(gtmpl)
    Nps_H = len(H_inds[0])
    Nps_g = len(g_inds[0])
    Nps = Nps_g + Nm + Nps_H
    
    tmpH = jnp.zeros((Nm,Nm))
    tmpg = jnp.zeros((Ne,Nm))
    λmin,λmax = (ω[0],ω[-1]) if λlims is None else λlims

    def Hκg_to_ps(H,κ,g):
        return jnp.hstack((g[g_inds],κ,H[H_inds]))

    @jit
    def ps_to_Hκg(ps):
        gps,κ,Hps = jnp.split(ps,[Nps_g,Nps_g+Nm])
        g  = jax.ops.index_update(tmpg,g_inds,gps)
        Hu = jax.ops.index_update(tmpH,H_inds,Hps)
        H = Hu + jnp.tril(Hu.T,-1)
        return H,κ,g

    @jit
    def Jfun(ω,ps):
        H,κ,g = ps_to_Hκg(ps)
        Heff = H-0.5j*jnp.diag(κ)
        JJ = Jmod(ω,Heff,g)
        return JJ

    @jit
    def err(ps):
        Jf = Jfun(ω,ps)
        if fitlog:
            return jnp.linalg.norm(jnp.log(Jf)-jnp.log(J))
        else:
            return jnp.linalg.norm(Jf-J)

    grad_err = jit(grad(err))

    def nlopt_f(ps,grad):
        if grad.size>0:
            grad[...] = grad_err(ps)
        return float(err(ps))
    
    @jit
    def f_constraints(ps):
        "constraint function that forces eigenvalues to be within the range [λmin,λmax]"
        H,κ,g = ps_to_Hκg(ps)
        λs = jnp.linalg.eigvalsh(H)
        # nlopt enforces constraint functions to be smaller than 0
        return jnp.hstack((λs-λmax,λmin-λs))

    jac_constraints = jit(jacobian(f_constraints))

    def nlopt_constraints(result,ps,grad):
        result[...] = f_constraints(ps)
        if grad.size > 0:
            grad[...] = jac_constraints(ps)

    opt = nlopt.opt(nlopt.LD_MMA,Nps)
    opt.set_min_objective(nlopt_f)
    opt.add_inequality_mconstraint(nlopt_constraints, np.zeros(2*Nm))
    opt.set_ftol_rel(1e-5)
    
    # add members to opt to have everything in a single object
    opt.Hκg_to_ps = Hκg_to_ps
    opt.ps_to_Hκg = ps_to_Hκg
    opt.Jfun = Jfun
    opt.obj_fun = nlopt_f
    
    return opt

def make_objfun(ω,J,Nm,λlims=None,fitlog=False):
    Ne = jnp.asarray(J).shape[-1]
    Htmpl = jnp.ones((Nm,Nm))
    gtmpl = jnp.ones((Ne,Nm))
    return make_objfun_shaped(ω, J, Htmpl, gtmpl, λlims, fitlog)

