"""
kgpy.optics is a package designed for simulating optical systems.
"""

__all__ = ['ZemaxCompatible', 'OCC_Compatible', 'Rays', 'Material', 'Aperture', 'Surface', 'System']

from .zemax_compatible import ZemaxCompatible
from .occ_compatible import OCC_Compatible
from .rays import Rays
from .aperture import Aperture
from .material import Material
from .surface import Surface
from .system import System
