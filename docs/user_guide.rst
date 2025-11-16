User Guide
==========

This guide provides detailed information on using spectral_density_fit for various fitting scenarios.

The Few-Mode Model
------------------

The package fits an arbitrary spectral density J(ω) with a few-mode model described by:

- A real symmetric Hamiltonian ``H``
- Decay rates ``κ`` (positive real values)
- A coupling matrix ``g``

The effective Hamiltonian is ``H_eff = H - 0.5j * diag(κ)`` (complex symmetric).

The spectral density is then computed as:

.. math::

    J(\omega) = \frac{1}{\pi} \text{Im}\left[g^\dagger \frac{1}{H_{\text{eff}} - \omega I} g\right]

Basic Fitting
-------------

Single Emitter
~~~~~~~~~~~~~~

For a single emitter, provide a 1D array of spectral density values:

.. code-block:: python

    import jax
    import numpy as np
    from spectral_density_fit import spectral_density_fitter

    jax.config.update("jax_enable_x64", True)

    ω = np.linspace(-5, 5, 200)
    J = ...  # Your spectral density data (shape: (Nω,))
    
    # Fit with Nm modes
    Nm = 5
    fitter = spectral_density_fitter(ω, J, Nm)

Multiple Emitters
~~~~~~~~~~~~~~~~~

For multiple emitters, provide a 3D array where J[i, j, k] represents the cross-spectral density between emitters i and j at frequency ω[k]:

.. code-block:: python

    Ne = 3  # Number of emitters
    J = np.zeros((Ne, Ne, len(ω)), dtype=complex)
    
    # Fill diagonal with spectral densities
    J[0, 0, :] = ...  # Spectral density for emitter 1
    J[1, 1, :] = ...  # Spectral density for emitter 2
    J[2, 2, :] = ...  # Spectral density for emitter 3
    
    # Fill off-diagonal with cross-spectral densities (if any)
    J[0, 1, :] = ...
    J[1, 0, :] = J[0, 1, :].conj()  # Hermitian symmetry
    
    fitter = spectral_density_fitter(ω, J, Nm)

Advanced Features
-----------------

Custom Templates
~~~~~~~~~~~~~~~~

You can restrict which elements of H and g can be nonzero using templates:

.. code-block:: python

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

Eigenvalue Constraints
~~~~~~~~~~~~~~~~~~~~~~

By default, the fitter constrains eigenvalues of H to be within the frequency range [ω.min(), ω.max()]. You can customize this:

.. code-block:: python

    # Custom eigenvalue limits
    fitter = spectral_density_fitter(ω, J, Nm, λlims=(-10, 10))
    
    # Disable eigenvalue constraints
    fitter = spectral_density_fitter(ω, J, Nm, λlims=False)

Logarithmic Fitting
~~~~~~~~~~~~~~~~~~~

For spectral densities spanning many orders of magnitude, logarithmic fitting can be more effective:

.. code-block:: python

    # Note: Only works for single emitter (Ne=1)
    fitter = spectral_density_fitter(ω, J, Nm, fitlog=True)

This minimizes the error in log-space: ``||log(J_fit) - log(J_target)||`` instead of ``||J_fit - J_target||``.

GPU Acceleration
~~~~~~~~~~~~~~~~

The package automatically uses GPU if available. You can control the device and algorithm:

.. code-block:: python

    # Force CPU with diagonalization (default on CPU)
    fitter = spectral_density_fitter(ω, J, Nm, diagonalize=True, device=jax.devices("cpu")[0])
    
    # Use GPU with direct inversion (default on GPU)
    if len(jax.devices("gpu")) > 0:
        fitter = spectral_density_fitter(ω, J, Nm, diagonalize=False, device=jax.devices("gpu")[0])

.. note::

    GPU diagonalization of non-Hermitian matrices is not supported in JAX. On GPU, the fitter uses direct matrix inversion (``Jmod_naive``), which is slower per iteration but benefits from GPU speedup.

Optimization
------------

Choosing Initial Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Good initial conditions are important for convergence:

.. code-block:: python

    # Random initialization (small values)
    ps0 = np.random.normal(size=fitter.Nps) * 0.1
    
    # If you have a good guess for H, κ, g:
    H_guess = ...
    κ_guess = ...
    g_guess = ...
    ps0 = fitter.Hκg_to_ps(H_guess, κ_guess, g_guess)

Monitoring Progress
~~~~~~~~~~~~~~~~~~~

You can add a callback to monitor optimization progress:

.. code-block:: python

    import nlopt
    
    def callback(ps, obj_val):
        print(f"Objective value: {obj_val:.6e}")
        # Return > 0 to stop optimization
        return 0
    
    fitter = spectral_density_fitter(ω, J, Nm)
    # Note: Not all NLopt algorithms support callbacks

Changing Optimization Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use different NLopt algorithms:

.. code-block:: python

    import nlopt
    
    # Use BOBYQA (derivative-free)
    fitter = spectral_density_fitter(ω, J, Nm, algorithm=nlopt.LN_BOBYQA)
    
    # Use SLSQP (gradient-based)
    fitter = spectral_density_fitter(ω, J, Nm, algorithm=nlopt.LD_SLSQP)

The default algorithm is ``nlopt.LD_CCSAQ``, which typically provides good performance.

Adjusting Tolerances
~~~~~~~~~~~~~~~~~~~~

You can adjust stopping criteria:

.. code-block:: python

    fitter = spectral_density_fitter(ω, J, Nm)
    
    # Set relative tolerance on objective function
    fitter.set_ftol_rel(1e-6)  # default is 1e-5
    
    # Set absolute tolerance
    fitter.set_ftol_abs(1e-8)
    
    # Set maximum number of evaluations
    fitter.set_maxeval(10000)
    
    # Stop when objective value reaches threshold
    fitter.set_stopval(1e-6)

Computing Spectral Densities
-----------------------------

There are two methods for computing spectral densities directly:

Using Jmod (Diagonalization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spectral_density_fit import Jmod
    import jax.numpy as jnp
    
    # Create effective Hamiltonian
    H = ...  # Complex symmetric matrix
    κ = ...  # Decay rates
    g = ...  # Coupling
    Heff = H - 0.5j * jnp.diag(κ)
    
    # Compute spectral density
    J = Jmod(ω, Heff, g)

This method diagonalizes ``Heff`` and is more efficient but only works on CPU.

Using Jmod_naive (Direct Inversion)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spectral_density_fit import Jmod_naive
    
    J = Jmod_naive(ω, Heff, g)

This method solves a linear equation at each frequency and is compatible with GPUs.

Troubleshooting
---------------

Poor Convergence
~~~~~~~~~~~~~~~~

If the optimization doesn't converge well:

1. Try different initial conditions
2. Increase the number of modes (Nm)
3. Try a different optimization algorithm
4. Use eigenvalue constraints if not already enabled
5. Check that JAX is using 64-bit precision

Slow Performance
~~~~~~~~~~~~~~~~

If optimization is slow:

1. Use GPU if available (automatic)
2. Reduce the number of frequency points if possible
3. Consider using ``diagonalize=True`` on CPU for better per-iteration speed
4. Use coarser tolerances for initial exploration

Memory Issues
~~~~~~~~~~~~~

If you encounter memory issues:

1. Reduce the number of frequency points
2. Reduce the number of modes (Nm)
3. Use CPU instead of GPU (GPU uses more memory for intermediate computations)
