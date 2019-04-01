from .surface import *
from .component import *
from .baffle import *
from .system import *
from kgpy.optics.surface.aperture import *

import sys
print(sys.platform)
if sys.platform == 'win32':
    from .zemax import ZmxSurface, ZmxSystem
