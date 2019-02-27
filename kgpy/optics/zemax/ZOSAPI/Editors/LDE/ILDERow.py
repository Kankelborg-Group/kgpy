
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

    def AvailableSurfaceTypes(self) -> List['ZOSAPI.Editors.LDE.SurfaceType']:
        pass

    def ChangeType(self, settings: 'ZOSAPI.Editors.LDE.ISurfaceTypeSettings') -> bool:
        pass

    def GetCellAt(self, pos: int) -> 'ZOSAPI.Editors.IEditorCell':
        pass

    def GetSurfaceCell(self, col: 'ZOSAPI.Editors.LDE.SurfaceColumn') -> 'ZOSAPI.Editors.IEditorCell':
        pass

    def GetSurfaceTypeSettings(self, type: 'ZOSAPI.Editors.LDE.SurfaceType'
                               ) -> 'ZOSAPI.Editors.LDE.ISurfaceTypeSettings':
        pass
