
# from . import zemax

from .system import System

import sys
if sys.platform == 'win32':
    from .zemax import ZmxSurface, ZmxSystem
