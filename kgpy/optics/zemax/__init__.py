from . import system
from . import ZOSAPI

import sys
if sys.platform == 'win32':
    from .system import System
