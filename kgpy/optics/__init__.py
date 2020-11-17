"""
kgpy.optics is a package designed for simulating optical systems.
"""

__all__ = [
    'Distortion',
    'Vignetting',
    'Rays',
    'RaysList',
    'sag',
    'aperture',
    'material',
    'rulings',
    'Surface',
    'SurfaceList',
    'component',
    'System',
]

from ._distortion import Distortion
from ._vignetting import Vignetting
from ._rays import Rays
from ._rays_list import RaysList
from . import sag
from . import aperture
from . import material
from . import rulings
from ._surface import Surface
from ._surface_list import SurfaceList
from . import component
from ._system import System

Distortion.__module__ = __name__
Vignetting.__module__ = __name__
Rays.__module__ = __name__
RaysList.__module__ = __name__
Surface.__module__ = __name__
SurfaceList.__module__ = __name__
System.__module__ = __name__
