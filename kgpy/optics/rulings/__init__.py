"""
Rulings are tiny grooves etched into optics to make diffraction gratings.
This package allows for the simulation of various groove profiles.
"""
__all__ = ['Rulings', 'ConstantDensity', 'CubicPolyDensity']

from ._rulings import Rulings
from ._constant_density import ConstantDensity
from ._cubic_poly_density import CubicPolyDensity
