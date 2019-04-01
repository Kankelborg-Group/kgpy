import sys
if sys.platform == 'win32':
    from kgpy.optics.zemax.surface.aperture import *
    from .surface import *
    from .system import *
