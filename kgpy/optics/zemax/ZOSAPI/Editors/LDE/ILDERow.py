
from typing import List
from kgpy.optics.zemax import ZOSAPI

__all__ = ['ILDERow']


class ILDERow:

    ApertureData = None         # type: ZOSAPI.Editors.LDE.ILDEApertureData
    Comment = None              # type: str
    Conic = None                # type: float
    Editor = None               # type: ZOSAPI.Editors.LDE.ILensDataEditor
    IsActive = None             # type: bool
    IsImage = None              # type: bool
    IsObject = None             # type: bool
    IsStop = None               # type: bool
    Material = None             # type: str
    Radius = None               # type: float
    RowIndex = None             # type: int
    SurfaceNumber = None        # type: int
    Thickness = None            # type: float
    TiltDecenterData = None     # type: ZOSAPI.Editors.LDE.ILDETiltDecenterData
    Type = None                 # type: ZOSAPI.Editors.LDE.SurfaceType

    def AvailableSurfaceTypes(self) -> List[int]:
        pass

    def ChangeType(self, settings: ZOSAPI.Editors.LDE.ISurfaceTypeSettings):
        pass