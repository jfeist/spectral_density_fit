# Documentation for spectral_density_fit

This directory contains the Sphinx documentation for the spectral_density_fit package.

## Online Documentation

📖 **The documentation is available online at: https://jfeist.github.io/spectral_density_fit/**

The documentation is automatically built and deployed from the `main` branch via GitHub Actions.

## Building the Documentation Locally

The project uses Sphinx for documentation. Documentation dependencies are managed as a dependency group (PEP 735).

**Requirements:** Python 3.9+ and uv (recommended)

```bash
# Sync dependencies including the docs group
uv sync --group docs

# Build the documentation
uv run sphinx-build -b html docs docs/_build/html

# Or use make
cd docs
uv run make html
```

The built documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser to view it.

## Documentation Structure

- `index.rst` - Main documentation page with table of contents
- `installation.rst` - Installation instructions
- `quickstart.rst` - Quick start guide with a basic example
- `user_guide.rst` - Comprehensive user guide covering all features
- `api_reference.rst` - API documentation (auto-generated from docstrings)
- `examples.rst` - Detailed examples for various use cases
- `conf.py` - Sphinx configuration

## Cleaning Build Files

To clean the build directory:

```bash
make clean
```

## Note

The documentation uses the NumPy docstring format. When adding or updating docstrings in the source code, follow the NumPy style guide:
https://numpydoc.readthedocs.io/en/latest/format.html
