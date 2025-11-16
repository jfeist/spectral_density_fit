Quick Start
===========

This page provides a quick introduction to using spectral_density_fit.

Basic Example
-------------

Here's a simple example that fits a Lorentzian spectral density:

.. code-block:: python

    import jax
    import numpy as np
    from spectral_density_fit import spectral_density_fitter

    # Enable 64-bit precision (required for accurate fitting)
    jax.config.update("jax_enable_x64", True)

    # Create a frequency array
    ω = np.linspace(-3, 3, 100)

    # Define a target Lorentzian spectral density
    γ = 0.1
    ω0 = 0.5
    J_target = γ / ((ω - ω0)**2 + γ**2)

    # Fit with 3 modes
    Nm = 3
    fitter = spectral_density_fitter(ω, J_target, Nm)

    # Initial guess
    ps0 = np.random.normal(size=fitter.Nps) * 0.1

    # Run optimization
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

The effective Hamiltonian used in the spectral density calculation is ``H_eff = H - 0.5j * diag(κ)``, which is complex symmetric.

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
