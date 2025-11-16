Examples
========

This page provides detailed examples for various use cases.

Example 1: Fitting a Single Lorentzian
---------------------------------------

This example demonstrates fitting a simple Lorentzian spectral density.

.. code-block:: python

    import jax
    import numpy as np
    import matplotlib.pyplot as plt
    from spectral_density_fit import spectral_density_fitter

    # Enable 64-bit precision
    jax.config.update("jax_enable_x64", True)

    # Define frequency range and target spectral density
    ω = np.linspace(-5, 5, 201)
    γ = 0.2
    ω0 = 1.0
    J_target = γ / ((ω - ω0)**2 + γ**2)

    # Fit with 3 modes
    Nm = 3
    fitter = spectral_density_fitter(ω, J_target, Nm)

    # Initialize with small random values
    np.random.seed(42)
    ps0 = np.random.normal(size=fitter.Nps) * 0.1

    # Optimize
    ps_opt = fitter.optimize(ps0)

    # Get fitted spectral density
    J_fit = fitter.Jfun(ω, ps_opt)

    # Extract parameters
    H, κ, g = fitter.ps_to_Hκg(ps_opt)

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(ω, J_target, 'k-', linewidth=2, label='Target')
    plt.plot(ω, J_fit[0, 0, :], 'r--', linewidth=2, label='Fit')
    plt.xlabel('Frequency ω')
    plt.ylabel('J(ω)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Spectral Density')

    plt.subplot(1, 2, 2)
    plt.semilogy(ω, J_target, 'k-', linewidth=2, label='Target')
    plt.semilogy(ω, J_fit[0, 0, :], 'r--', linewidth=2, label='Fit')
    plt.xlabel('Frequency ω')
    plt.ylabel('J(ω)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Spectral Density (log scale)')

    plt.tight_layout()
    plt.show()

    # Print parameters
    print(f"Fit error: {np.linalg.norm(J_fit - J_target[None, None, :]):.6e}")
    print(f"\\nEffective Hamiltonian H:\\n{H}")
    print(f"\\nDecay rates κ:\\n{κ}")
    print(f"\\nCoupling g:\\n{g}")

Example 2: Multiple Lorentzians
--------------------------------

Fitting a spectral density with multiple peaks.

.. code-block:: python

    import jax
    import numpy as np
    import matplotlib.pyplot as plt
    from spectral_density_fit import spectral_density_fitter

    jax.config.update("jax_enable_x64", True)

    # Define frequency range
    ω = np.linspace(-10, 10, 401)

    # Create a spectral density with three Lorentzian peaks
    J_target = (
        0.3 / ((ω - 2.0)**2 + 0.1**2) +
        0.5 / ((ω + 0.0)**2 + 0.2**2) +
        0.4 / ((ω + 3.0)**2 + 0.15**2)
    )

    # Fit with 6 modes
    Nm = 6
    fitter = spectral_density_fitter(ω, J_target, Nm)

    # Initialize
    np.random.seed(42)
    ps0 = np.random.normal(size=fitter.Nps) * 0.1

    # Optimize
    ps_opt = fitter.optimize(ps0)
    J_fit = fitter.Jfun(ω, ps_opt)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ω, J_target, 'k-', linewidth=2, label='Target')
    plt.plot(ω, J_fit[0, 0, :], 'r--', linewidth=2, label='Fit')
    plt.xlabel('Frequency ω')
    plt.ylabel('J(ω)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Multi-Peak Spectral Density')
    plt.show()

    error = np.linalg.norm(J_fit - J_target[None, None, :])
    print(f"Fit error: {error:.6e}")

Example 3: Ohmic Spectral Density with Cutoff
----------------------------------------------

Fitting an Ohmic spectral density with exponential cutoff.

.. code-block:: python

    import jax
    import numpy as np
    import matplotlib.pyplot as plt
    from spectral_density_fit import spectral_density_fitter

    jax.config.update("jax_enable_x64", True)

    # Define frequency range (positive frequencies only)
    ω = np.linspace(0.01, 10, 501)

    # Ohmic spectral density with exponential cutoff
    α = 0.1  # Coupling strength
    ωc = 2.0  # Cutoff frequency
    J_target = α * ω * np.exp(-ω / ωc)

    # Fit with more modes for complex shape
    Nm = 8
    
    # Use logarithmic fitting for better accuracy across orders of magnitude
    fitter = spectral_density_fitter(ω, J_target, Nm, fitlog=True)

    # Initialize
    np.random.seed(42)
    ps0 = np.random.normal(size=fitter.Nps) * 0.1

    # Optimize
    ps_opt = fitter.optimize(ps0)
    J_fit = fitter.Jfun(ω, ps_opt)

    # Plot in both linear and log scale
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(ω, J_target, 'k-', linewidth=2, label='Target')
    axes[0].plot(ω, J_fit[0, 0, :], 'r--', linewidth=2, label='Fit')
    axes[0].set_xlabel('Frequency ω')
    axes[0].set_ylabel('J(ω)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Linear Scale')

    axes[1].semilogy(ω, J_target, 'k-', linewidth=2, label='Target')
    axes[1].semilogy(ω, J_fit[0, 0, :], 'r--', linewidth=2, label='Fit')
    axes[1].set_xlabel('Frequency ω')
    axes[1].set_ylabel('J(ω)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Log Scale')

    plt.tight_layout()
    plt.show()

    error = np.linalg.norm(J_fit - J_target[None, None, :])
    print(f"Fit error: {error:.6e}")

Example 4: Multiple Emitters
-----------------------------

Fitting spectral densities for multiple emitters with cross-correlations.

.. code-block:: python

    import jax
    import numpy as np
    import matplotlib.pyplot as plt
    from spectral_density_fit import spectral_density_fitter, Jmod
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    # First, create a target with known few-mode parameters
    Ne = 2  # Two emitters
    Nm = 4  # Four modes
    ω = np.linspace(-5, 5, 201)

    # Create random but reasonable parameters
    np.random.seed(42)
    H = np.random.normal(size=(Nm, Nm)) * 0.5
    H = H + H.T  # Make symmetric
    κ = np.random.uniform(0.1, 0.5, Nm)
    g = np.random.normal(size=(Ne, Nm)) * 0.3

    # Create effective Hamiltonian
    Heff = H - 0.5j * np.diag(κ)

    # Compute target spectral density
    J_target = Jmod(ω, Heff, g)

    # Now fit it
    fitter = spectral_density_fitter(ω, J_target, Nm)

    # Initialize with perturbed values
    H_init = H + np.random.normal(size=H.shape) * 0.1
    H_init = H_init + H_init.T
    κ_init = κ + np.random.normal(size=κ.shape) * 0.05
    κ_init = np.abs(κ_init)
    g_init = g + np.random.normal(size=g.shape) * 0.05
    ps0 = fitter.Hκg_to_ps(H_init, κ_init, g_init)

    # Optimize
    ps_opt = fitter.optimize(ps0)
    J_fit = fitter.Jfun(ω, ps_opt)

    # Plot all components
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # J_11
    axes[0, 0].plot(ω, J_target[0, 0, :].real, 'k-', linewidth=2, label='Target')
    axes[0, 0].plot(ω, J_fit[0, 0, :].real, 'r--', linewidth=2, label='Fit')
    axes[0, 0].set_xlabel('Frequency ω')
    axes[0, 0].set_ylabel('J₁₁(ω)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title('Spectral Density: Emitter 1')

    # J_22
    axes[0, 1].plot(ω, J_target[1, 1, :].real, 'k-', linewidth=2, label='Target')
    axes[0, 1].plot(ω, J_fit[1, 1, :].real, 'r--', linewidth=2, label='Fit')
    axes[0, 1].set_xlabel('Frequency ω')
    axes[0, 1].set_ylabel('J₂₂(ω)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title('Spectral Density: Emitter 2')

    # J_12 (real part)
    axes[1, 0].plot(ω, J_target[0, 1, :].real, 'k-', linewidth=2, label='Target')
    axes[1, 0].plot(ω, J_fit[0, 1, :].real, 'r--', linewidth=2, label='Fit')
    axes[1, 0].set_xlabel('Frequency ω')
    axes[1, 0].set_ylabel('Re[J₁₂(ω)]')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title('Cross Spectral Density: Real Part')

    # J_12 (imaginary part)
    axes[1, 1].plot(ω, J_target[0, 1, :].imag, 'k-', linewidth=2, label='Target')
    axes[1, 1].plot(ω, J_fit[0, 1, :].imag, 'r--', linewidth=2, label='Fit')
    axes[1, 1].set_xlabel('Frequency ω')
    axes[1, 1].set_ylabel('Im[J₁₂(ω)]')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title('Cross Spectral Density: Imaginary Part')

    plt.tight_layout()
    plt.show()

    error = np.linalg.norm(J_fit - J_target)
    print(f"Fit error: {error:.6e}")

Example 5: Using Custom Templates
----------------------------------

Restricting the structure of the Hamiltonian and coupling.

.. code-block:: python

    import jax
    import numpy as np
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from spectral_density_fit import spectral_density_fitter

    jax.config.update("jax_enable_x64", True)

    # Frequency range
    ω = np.linspace(-5, 5, 201)

    # Target spectral density (single emitter)
    J_target = 0.2 / ((ω - 1.0)**2 + 0.1**2) + 0.3 / ((ω + 1.5)**2 + 0.15**2)

    # Create templates
    Nm = 4
    Ne = 1

    # Force H to be diagonal (uncoupled modes)
    Htmpl = jnp.eye(Nm)

    # All couplings allowed
    gtmpl = jnp.ones((Ne, Nm))

    # Fit with templates
    fitter = spectral_density_fitter(ω, J_target, (Htmpl, gtmpl))

    # Initialize
    np.random.seed(42)
    ps0 = np.random.normal(size=fitter.Nps) * 0.1

    # Optimize
    ps_opt = fitter.optimize(ps0)
    J_fit = fitter.Jfun(ω, ps_opt)

    # Extract parameters
    H, κ, g = fitter.ps_to_Hκg(ps_opt)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ω, J_target, 'k-', linewidth=2, label='Target')
    plt.plot(ω, J_fit[0, 0, :], 'r--', linewidth=2, label='Fit')
    plt.xlabel('Frequency ω')
    plt.ylabel('J(ω)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Fit with Diagonal Hamiltonian')
    plt.show()

    print("Hamiltonian (should be diagonal):")
    print(H)
    print(f"\\nNumber of fit parameters: {fitter.Nps}")
    print(f"(Reduced from {Nm*(Nm+1)//2 + Nm + Ne*Nm} without template)")

Example 6: Direct Spectral Density Calculation
-----------------------------------------------

Computing spectral densities directly without fitting.

.. code-block:: python

    import jax
    import numpy as np
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from spectral_density_fit import Jmod, Jmod_naive

    jax.config.update("jax_enable_x64", True)

    # Define parameters
    Nm = 3
    Ne = 1
    ω = np.linspace(-5, 5, 201)

    # Create effective Hamiltonian
    H = jnp.array([
        [0.0, 0.5, 0.0],
        [0.5, 1.0, 0.3],
        [0.0, 0.3, -1.0]
    ])
    κ = jnp.array([0.1, 0.2, 0.15])
    g = jnp.array([[0.3, 0.4, 0.2]])

    Heff = H - 0.5j * jnp.diag(κ)

    # Compute using both methods
    J1 = Jmod(ω, Heff, g)
    J2 = Jmod_naive(ω, Heff, g)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ω, J1[0, 0, :], 'b-', linewidth=2, label='Jmod (diagonalization)')
    plt.plot(ω, J2[0, 0, :], 'r--', linewidth=2, label='Jmod_naive (direct)')
    plt.xlabel('Frequency ω')
    plt.ylabel('J(ω)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Spectral Density from Few-Mode Parameters')
    plt.show()

    # Check they match
    print(f"Difference between methods: {np.max(np.abs(J1 - J2)):.6e}")
