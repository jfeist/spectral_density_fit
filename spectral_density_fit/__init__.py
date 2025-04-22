"""Fit arbitrary spectral densities with a few-mode model

Implements the procedure from I. Medina, F. J. García-Vidal, A. I.
Fernández-Domínguez, and J. Feist, Phys. Rev. Lett. 126, 093601 (2021),
https://doi.org/10.1103/PhysRevLett.126.093601
"""

__version__ = "0.2.1"
__all__ = ["spectral_density_fitter"]

from .spectral_densities import Jmod, Jmod_naive
from .fitters import spectral_density_fitter
