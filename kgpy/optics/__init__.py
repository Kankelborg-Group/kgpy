"""
kgpy.optics is a package designed for simulating optical systems.
"""

__all__ = [
    'Rays', 'RaysList',
    'aperture', 'Aperture',
    'material', 'Material',
    'surface', 'Surface', 'SurfaceList',
    'component',
    'System',
]

from .rays import Rays, RaysList
from . import aperture
from .aperture import Aperture
from . import material
from .material import Material
from . import surface
from .surface import Surface, SurfaceList
from . import component
from .system import System
