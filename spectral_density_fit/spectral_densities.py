import jax
import jax.numpy as jnp
from jax import jit
from functools import partial


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
