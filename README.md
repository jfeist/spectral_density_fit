# Spectral density fitter for few-mode quantization
This package can be used to perform few-mode quantization for multiple emitters with arbitrary spectral densities as presented in 

1. Few-Mode Field Quantization of Arbitrary Electromagnetic Spectral Densities, I. Medina, F. J. García-Vidal, A. I. Fernández-Domínguez, and J. Feist, [Phys. Rev. Lett. 126, 093601 (2021)](https://doi.org/10.1103/PhysRevLett.126.093601)
2. Few-mode field quantization for multiple emitters, M. Sánchez-Barquilla, F. J. García-Vidal, A. I. Fernández-Domínguez, and J. Feist, [Nanophotonics 11, 4363 (2022)](https://doi.org/10.1515/nanoph-2021-0795)

It is currently functional, but documentation is missing. To install, run 
```
pip install git+https://github.com/jfeist/spectral_density_fit
```
NOTE that `spectral_density_fit` recently changed from forcing 64-bit mode for jax (which the code uses to get the gradients of the function to minimize in the fit, and to support running on GPUs) to not touching the configuration automatically. In practice, it is likely that you want to use 64-bit precision for the fitting, to do so, you need to run, **before** any other calculations, the following:
```python
import jax
jax.config.update("jax_enable_x64", True)
```

If you want to develop the package, we recommend using
[uv](https://docs.astral.sh/uv) to manage the project and dependencies. In
particular, after checking out the git repository, you can simply run `uv run
pytest` to run the tests, and `uv run ruff check` or `uv run ruff format` to
lint and format the code.