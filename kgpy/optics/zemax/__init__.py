
from . import surface
from . import system
from . import ZOSAPI

import sys
if sys.platform == 'win32':
    from .surface import ZmxSurface
    from .system import *
