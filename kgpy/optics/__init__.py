from .surface import *
from .component import *
from .baffle import *
from .system import *

import sys
if sys.platform is 'win32':
    from .zemax import *