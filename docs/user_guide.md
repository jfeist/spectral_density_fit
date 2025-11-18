# User Guide

This guide provides comprehensive information on using spectral_density_fit for various fitting scenarios.

## Getting Started

### Configuring JAX Precision

:::{important}
The package requires 64-bit precision for accurate fitting. You must configure JAX **before** using the package:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

Without this configuration, you will receive a runtime warning, and fitting accuracy may be reduced.
:::

## The Few-Mode Model

The package fits an arbitrary spectral density $J(\omega)$ with a few-mode model described by:

- A real symmetric Hamiltonian `H` (of size `Nm x Nm`)
- Decay rates `κ` (positive real vector of size `Nm`)
- A coupling matrix `g` (of size `Ne x Nm`)

The complex symmetric effective Hamiltonian is $H_{\text{eff}} = H - \frac{i}{2} \mathrm{diag}(\kappa)$.

The spectral density is then computed as:

$$
J(\omega) = \frac{1}{\pi} g^\dagger \operatorname{Im}\left[\frac{1}{H_{\text{eff}} - \omega I}\right] g
$$

## Quick Start Example

Here's a simple example that demonstrates the basic workflow for fitting a Lorentzian spectral density:

```python
import jax
import numpy as np
from spectral_density_fit import spectral_density_fitter

# Enable 64-bit precision (required)
jax.config.update("jax_enable_x64", True)

# Define frequency range and target spectral density
ω = np.linspace(0, 5, 201)

# Single Lorentzian with proper spectral density form: g² κ / (2π) / ((ω - ω0)² + (κ/2)²)
# where g is the coupling strength, ω0 is the resonance frequency and κ is the decay rate
ω0 = 2.0  # resonance frequency
κ = 0.4   # decay rate
g = 0.3   # coupling strength
J_target = (g**2 * κ / (2 * np.pi)) / ((ω - ω0)**2 + (κ/2)**2)

# Fit with 1 mode (matching the single Lorentzian)
Nm = 1
fitter = spectral_density_fitter(ω, J_target, Nm)

# Initialize with reasonable guesses
H_init = np.array([[1.9]])   # Hamiltonian with resonance frequency
κ_init = np.array([0.3])     # Decay rate
g_init = np.array([[0.2]])   # Coupling strength
ps0 = fitter.Hκg_to_ps(H_init, κ_init, g_init)

# Optimize
ps_opt = fitter.optimize(ps0)

# Get fitted spectral density and extract parameters
J_fit = fitter.Jfun(ω, ps_opt)
H, κ_fit, g_fit = fitter.ps_to_Hκg(ps_opt)

# Compute the fit quality
error = np.linalg.norm(J_fit - J_target[None, None, :])
print(f"Fit error: {error:.6e}")
print(f"Hamiltonian H: {H}")
print(f"Decay rates κ: {κ_fit}")
print(f"Coupling g: {g_fit}")
```

### Understanding the Parameters

The fitter returns a parameter vector `ps_opt` that encodes the physical parameters:

- `H`: Real symmetric Hamiltonian of shape `(Nm, Nm)`
- `κ`: Decay rates (positive real values) of shape `(Nm,)`
- `g`: Coupling matrix of shape `(Ne, Nm)`

The effective Hamiltonian used in the spectral density calculation is $H_\mathrm{eff} = H - \frac{i}{2} \mathrm{diag}(\kappa)$, which is complex symmetric.

You can visualize the results with matplotlib:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(ω, J_target, 'k-', linewidth=2, label='Target')
plt.plot(ω, J_fit[0, 0, :], 'r--', linewidth=2, label='Fit')
plt.xlabel('Frequency ω')
plt.ylabel('Spectral density J(ω)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Multiple Emitters

For multiple emitters, provide a 3D array where J[i, j, k] represents the cross-spectral density between emitters i and j at frequency ω[k]. The number of emitters `Ne` is automatically inferred from the shape of `J`.

```python
Ne = 3  # Number of emitters
J = np.zeros((Ne, Ne, len(ω)))

# Fill with spectral densities
J[:, :, :] = ...  # Your spectral density data (shape: (Ne, Ne, Nω))

fitter = spectral_density_fitter(ω, J, Nm)
```

## Advanced Features

### Custom Templates

You can restrict which elements of H and g can be nonzero using templates. To do so, pass a tuple `(Htmpl, gtmpl)` instead of `Nm` when creating the fitter.
Here, `Htmpl` is a matrix of shape `(Nm, Nm)` indicating allowed nonzero elements in `H`, and `gtmpl` is a matrix of shape `(Ne, Nm)` for `g`.
In both templates, any non-zero value indicates an allowed element, while 0 indicates a disallowed element.

```python
import jax.numpy as jnp

Nm = 4
Ne = 2

# Allow only diagonal elements in H
Htmpl = jnp.eye(Nm)

# Allow only certain couplings
gtmpl = jnp.array([
    [1, 1, 0, 0],  # Emitter 1 couples to modes 1 and 2
    [0, 0, 1, 1],  # Emitter 2 couples to modes 3 and 4
])

fitter = spectral_density_fitter(ω, J, (Htmpl, gtmpl))
```

### Eigenvalue Constraints

By default, the fitter constrains eigenvalues of H to be within the frequency range [ω.min(), ω.max()]. You can customize this:

```python
# Custom eigenvalue limits
fitter = spectral_density_fitter(ω, J, Nm, λlims=(-10, 10))

# Disable eigenvalue constraints
fitter = spectral_density_fitter(ω, J, Nm, λlims=False)
```

### Logarithmic Fitting

For spectral densities spanning many orders of magnitude, logarithmic fitting can be more effective, especially if you need to capture both small and large features accurately.
This currently only works for the single-emitter case (Ne=1):

```python
# Note: Only works for single emitter (Ne=1)
fitter = spectral_density_fitter(ω, J, Nm, fitlog=True)
```

This minimizes the error in log-space: $||\log(J_\mathrm{fit}) - \log(J_\mathrm{target})||$ instead of $||J_\mathrm{fit} - J_\mathrm{target}||$.

### GPU Acceleration

The package automatically uses GPU if available. You can control the device and algorithm:

```python
# Force CPU with diagonalization (default on CPU)
fitter = spectral_density_fitter(ω, J, Nm, diagonalize=True, device=jax.devices("cpu")[0])

# Use GPU with direct inversion (default on GPU)
if len(jax.devices("gpu")) > 0:
    fitter = spectral_density_fitter(ω, J, Nm, diagonalize=False, device=jax.devices("gpu")[0])
```

:::{note}
GPU diagonalization of non-Hermitian matrices is not supported in JAX. On GPU, the fitter uses direct matrix inversion (`Jmod_naive`), which is slower per iteration but benefits from GPU speedup.
:::

## Optimization

### Choosing Initial Conditions

Good initial conditions are important for convergence:

```python
# Random initialization (small values)
ps0 = np.random.normal(size=fitter.Nps) * 0.1

# If you have a good guess for H, κ, g:
H_guess = ...
κ_guess = ...
g_guess = ...
ps0 = fitter.Hκg_to_ps(H_guess, κ_guess, g_guess)
```

### Changing Optimization Algorithm

You can use different NLopt algorithms:

```python
import nlopt

# Use BOBYQA (derivative-free)
fitter = spectral_density_fitter(ω, J, Nm, algorithm=nlopt.LN_BOBYQA)

# Use SLSQP (gradient-based)
fitter = spectral_density_fitter(ω, J, Nm, algorithm=nlopt.LD_SLSQP)
```

The default algorithm is `nlopt.LD_CCSAQ`, which typically provides good performance.

### Adjusting Tolerances

You can adjust stopping criteria:

```python
fitter = spectral_density_fitter(ω, J, Nm)

# Set relative tolerance on objective function
fitter.set_ftol_rel(1e-6)  # default is 1e-5

# Set absolute tolerance
fitter.set_ftol_abs(1e-8)

# Set maximum number of evaluations
fitter.set_maxeval(10000)

# Stop when objective value reaches threshold
fitter.set_stopval(1e-6)
```

## Computing Spectral Densities

There are two methods for computing spectral densities directly:

### Using Jmod (Diagonalization)

```python
from spectral_density_fit import Jmod
import jax.numpy as jnp

# Create effective Hamiltonian
H = ...  # Complex symmetric matrix
κ = ...  # Decay rates
g = ...  # Coupling
Heff = H - 0.5j * jnp.diag(κ)

# Compute spectral density
J = Jmod(ω, Heff, g)
```

This method diagonalizes `Heff` and is more efficient but only works on CPU.

### Using Jmod_naive (Direct Inversion)

```python
from spectral_density_fit import Jmod_naive

J = Jmod_naive(ω, Heff, g)
```

This method solves a linear equation at each frequency and is compatible with GPUs.

## Troubleshooting

### Poor Convergence

If the optimization doesn't converge well:

1. Try different initial conditions
2. Increase the number of modes (Nm)
3. Try a different optimization algorithm
4. Use eigenvalue constraints if not already enabled
5. Check that JAX is using 64-bit precision

### Slow Performance

If optimization is slow:

1. Use GPU if available (automatic)
2. Reduce the number of frequency points if possible
3. Consider using `diagonalize=True` on CPU for better per-iteration speed
4. Use coarser tolerances for initial exploration

### Memory Issues

If you encounter memory issues:

1. Reduce the number of frequency points
2. Reduce the number of modes (Nm)
3. Use CPU instead of GPU (GPU uses more memory for intermediate computations)
