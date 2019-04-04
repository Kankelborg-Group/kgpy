from .surface import Surface
from .component import Component
from .system import System

import sys
print(sys.platform)
if sys.platform == 'win32':
    from .zemax import ZmxSurface, ZmxSystem
