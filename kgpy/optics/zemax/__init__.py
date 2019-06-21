from kgpy.optics.zemax.system.configuration import surface
from . import system
from . import ZOSAPI

import sys
if sys.platform == 'win32':
    from kgpy.optics.zemax.system.configuration.surface import ZmxSurface
    from .system import *
