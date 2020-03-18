import typing as typ
from kgpy.optics.zemax import ZOSAPI

__all__ = ['ILDERow']

ISurfaceT = typ.TypeVar('ISurfaceT')


class ILDERow(typ.Generic[ISurfaceT]):

    ApertureData = None                     # type: ZOSAPI.Editors.LDE.ILDEApertureData
    ChipZone = None                         # type: float
    ChipZoneCell = None                     # type: ZOSAPI.Editors.IEditorCell
    Coating = None                          # type: str
    CoatingCell = None                      # type: ZOSAPI.Editors.IEditorCell
    CoatingData = None                      # type: ZOSAPI.Editors.LDE.ILDECoatingData
    Comment = None                          # type: str
    CommentCell = None                      # type: ZOSAPI.Editors.IEditorCell
    Conic = None                            # type: float
    ConicCell = None                        # type: ZOSAPI.Editors.IEditorCell
    CurrentTypeSettings = None              # type: ZOSAPI.Editors.LDE.ISurfaceTypeSettings
    DrawData = None                         # type: ZOSAPI.Editors.LDE.ILDEDrawData
    Editor = None                           # type: ZOSAPI.Editors.LDE.ILensDataEditor
    ImportData = None                       # type: ZOSAPI.Editors.LDE.ILDEImportData
    IsActive = None                         # type: bool
    IsImage = None                          # type: bool
    IsObject = None                         # type: bool
    IsStop = None                           # type: bool
    IsValidRow = None                       # type: bool
    Material = None                         # type: str
    MaterialCell = None                     # type: ZOSAPI.Editors.IEditorCell
    MechanicalSemiDiameter = None           # type: float
    MechanicalSemiDiameterCell = None       # type: ZOSAPI.Editors.IEditorCell
    PhysicalOpticsData = None               # type: ZOSAPI.Editors.LDE.ILDEPhysicalOpticsData
    Radius = None                           # type: float
    RadiusCell = None                       # type: ZOSAPI.Editors.IEditorCell
    RowIndex = None                         # type: int
    RowTypeName = None                      # type: str
    ScatteringData = None                   # type: ZOSAPI.Editors.LDE.ILDEScatteringData
    SemiDiameter = None                     # type: float
    SemiDiameterCell = None                 # type: ZOSAPI.Editors.IEditorCell
    SurfaceData = None                      # type: ISurfaceT
    SurfaceNumber = None                    # type: int
    TCE = None                              # type: float
    TCECell = None                          # type: ZOSAPI.Editors.IEditorCell
    Thickness = None                        # type: float
    ThicknessCell = None                    # type: ZOSAPI.Editors.IEditorCell
    TiltDecenterData = None                 # type: ZOSAPI.Editors.LDE.ILDETiltDecenterData
    Type = None                             # type: ZOSAPI.Editors.LDE.SurfaceType
    TypeData = None                         # type: ZOSAPI.Editors.LDE.ILDETypeData
    TypeName = None                         # type: str

    def AvailableSurfaceTypes(self) -> typ.List['ZOSAPI.Editors.LDE.SurfaceType']:
        pass

    def ChangeType(self, settings: 'ZOSAPI.Editors.LDE.ISurfaceTypeSettings') -> bool:
        pass

    def GetCellAt(self, pos: int) -> 'ZOSAPI.Editors.IEditorCell':
        pass

    def GetSurfaceCell(self, col: 'ZOSAPI.Editors.LDE.SurfaceColumn') -> 'ZOSAPI.Editors.IEditorCell':
        pass

    def GetSurfaceTypeSettings(
            self, type_: 'ZOSAPI.Editors.LDE.SurfaceType'
    ) -> 'ZOSAPI.Editors.LDE.ISurfaceTypeSettings':
        pass
