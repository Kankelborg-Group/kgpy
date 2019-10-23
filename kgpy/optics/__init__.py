
# from .system import System

import platform
if platform.system() == 'Windows':
    from . import zemax


