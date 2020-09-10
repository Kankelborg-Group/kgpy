"""
kgpy.optics is a package designed for simulating optical systems.
"""

__all__ = [
    'Distortion',
    'Vignetting',
    'Rays', 'RaysList',
    'sag', 'Sag',
    'aperture', 'Aperture',
    'material', 'Material',
    'rulings', 'Rulings',
    'surface', 'Surface', 'SurfaceList',
    'component', 'Component',
    'System',
]

from .distortion import Distortion
from .vignetting import Vignetting
from .rays import Rays, RaysList
from . import sag
from .sag import Sag
from . import aperture
from .aperture import Aperture
from . import material
from .material import Material
from . import rulings
from .rulings import Rulings
from . import surface
from .surface import Surface, SurfaceList
from . import component
from .component import Component
from .system import System
