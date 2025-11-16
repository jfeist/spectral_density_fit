# Spectral density fitter for few-mode quantization

This package can be used to perform few-mode quantization for multiple emitters with arbitrary spectral densities as presented in

1. Few-Mode Field Quantization of Arbitrary Electromagnetic Spectral Densities, I. Medina, F. J. García-Vidal, A. I. Fernández-Domínguez, and J. Feist, [Phys. Rev. Lett. 126, 093601 (2021)](https://doi.org/10.1103/PhysRevLett.126.093601)
2. Few-mode field quantization for multiple emitters, M. Sánchez-Barquilla, F. J. García-Vidal, A. I. Fernández-Domínguez, and J. Feist, [Nanophotonics 11, 4363 (2022)](https://doi.org/10.1515/nanoph-2021-0795)

## Documentation

**📖 [Read the documentation online](https://jfeist.github.io/spectral_density_fit/)**

The documentation is automatically built and deployed from the `main` branch. It includes:
- Installation instructions
- Quick start guide
- Comprehensive user guide
- API reference with detailed function documentation
- Multiple examples for various use cases

### Building Documentation Locally

To build the documentation locally, install the documentation dependencies and build with Sphinx:

**Using pip:**
```bash
pip install -e ".[docs]"
cd docs
make html
# Open docs/_build/html/index.html in your browser
```

**Using uv (recommended):**
```bash
uv sync --group docs
uv run sphinx-build -b html docs docs/_build/html
# Open docs/_build/html/index.html in your browser
```

## Installation

To install, run:
```bash
pip install git+https://github.com/jfeist/spectral_density_fit
```

## Quick Start

**Important:** `spectral_density_fit` requires 64-bit precision for accurate fitting. You must configure JAX **before** importing the package:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

### Basic Example

```python
import jax
import numpy as np
from spectral_density_fit import spectral_density_fitter

# Enable 64-bit precision (required)
jax.config.update("jax_enable_x64", True)

# Create frequency array and target spectral density
ω = np.linspace(-3, 3, 100)
J_target = 0.1 / (ω**2 + 0.1**2)  # Lorentzian

# Fit with 3 modes
fitter = spectral_density_fitter(ω, J_target, 3)

# Initial guess
ps0 = np.random.normal(size=fitter.Nps) * 0.1

# Optimize
ps_opt = fitter.optimize(ps0)

# Get fitted spectral density
J_fit = fitter.Jfun(ω, ps_opt)
```

For more examples and detailed usage, see the documentation in the `docs/` directory.

## Development

If you want to develop the package, we recommend using
[uv](https://docs.astral.sh/uv) to manage the project and dependencies. In
particular, after checking out the git repository, you can simply run `uv run
pytest` to run the tests, and `uv run ruff check` or `uv run ruff format` to
lint and format the code.