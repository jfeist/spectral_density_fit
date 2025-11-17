Quick Start
===========

This page provides a quick introduction to using spectral_density_fit.

Configuring JAX Precision
--------------------------

.. important::

    The package requires 64-bit precision for accurate fitting. You must configure JAX **before** using the package:

    .. code-block:: python

        import jax
        jax.config.update("jax_enable_x64", True)

    Without this configuration, you will receive a runtime warning, and fitting accuracy may be reduced.

Basic Example
-------------

Here's a simple example that fits a Lorentzian spectral density:

.. code-block:: python

    import jax
    import numpy as np
    from spectral_density_fit import spectral_density_fitter

    # Enable 64-bit precision
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
    H_init = np.array([[1.9]])   # Coupling matrix with resonance frequency
    κ_init = np.array([0.3])     # Decay rate
    g_init = np.array([[0.2]])   # Coupling strength
    ps0 = fitter.Hκg_to_ps(H_init, κ_init, g_init)

    # Optimize
    ps_opt = fitter.optimize(ps0)

    # Get fitted spectral density
    J_fit = fitter.Jfun(ω, ps_opt)

    # Compute the fit quality
    error = np.linalg.norm(J_fit - J_target[None, None, :])
    print(f"Fit error: {error:.6f}")

Understanding the Output
-------------------------

The fitter returns a parameter vector ``ps_opt`` that can be converted back to physical parameters:

.. code-block:: python

    # Extract physical parameters
    H, κ, g = fitter.ps_to_Hκg(ps_opt)
    
    print(f"Effective Hamiltonian shape: {H.shape}")  # (Nm, Nm)
    print(f"Decay rates shape: {κ.shape}")            # (Nm,)
    print(f"Coupling shape: {g.shape}")               # (Ne, Nm)

Where:

- ``H`` is the real symmetric Hamiltonian matrix
- ``κ`` are the decay rates (positive real values)
- ``g`` is the coupling matrix

The effective Hamiltonian used in the spectral density calculation is :math:`H_\mathrm{eff} = H - \frac{i}{2} \mathrm{diag}(\kappa)`, which is complex symmetric.

Plotting the Results
--------------------

.. code-block:: python

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(ω, J_target, 'k-', linewidth=2, label='Target')
    plt.plot(ω, J_fit[0, 0, :], 'r--', linewidth=2, label='Fit')
    plt.xlabel('Frequency ω')
    plt.ylabel('Spectral density J(ω)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

Next Steps
----------

- See the :doc:`user_guide` for more detailed information on different fitting scenarios
- Check the :doc:`api_reference` for complete function and class documentation
- Explore :doc:`examples` for more complex use cases
