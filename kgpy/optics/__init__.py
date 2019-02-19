from .surface import *
from .component import *
from .baffle import *
from .system import *

import sys
print(sys.platform)
if sys.platform == 'win32':
    from .zemax import *
