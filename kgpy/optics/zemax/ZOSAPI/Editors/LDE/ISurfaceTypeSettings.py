
from kgpy.optics.zemax import ZOSAPI


class ISurfaceTypeSettings:

    Filename = None             # type: str
    IsValid = None              # type: bool
    RequiresFile = None         # type: bool
    Type = None                 # type: ZOSAPI.

    def GetFileNames(self):
        pass