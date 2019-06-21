
# from . import zemax

from kgpy.optics.system.configuration.surface import Surface
from kgpy.optics.system.configuration.component.component import Component
from .system import System

import sys
if sys.platform == 'win32':
    from .zemax import ZmxSurface, ZmxSystem
