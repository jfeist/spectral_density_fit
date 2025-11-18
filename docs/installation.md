# Installation

## Requirements

The package requires Python 3.9 or later, along with the following dependencies:

- numpy ≥ 1.15
- jax ≥ 0.3
- nlopt ≥ 2.7.0

## Installing from GitHub

To install the latest version directly from the GitHub repository:

```bash
pip install git+https://github.com/jfeist/spectral_density_fit
```

## Development Installation

If you want to develop the package, we recommend using [uv](https://docs.astral.sh/uv) to manage the project and dependencies:

1. Clone the repository:

   ```bash
   git clone https://github.com/jfeist/spectral_density_fit.git
   cd spectral_density_fit
   ```

2. Install dependencies and run tests:

   ```bash
   uv run pytest
   ```

3. Run linting and formatting:

   ```bash
   uv run ruff check
   uv run ruff format
   ```
