# Documentation for spectral_density_fit

This directory contains the Sphinx documentation for the spectral_density_fit package.

## Online Documentation

📖 **The documentation is available online at: https://jfeist.github.io/spectral_density_fit/**

The documentation is automatically built and deployed from the `main` branch via GitHub Actions.

## Building the Documentation Locally

To build the HTML documentation locally:

```bash
pip install sphinx sphinx-rtd-theme
cd docs
make html
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
