
from typing import Tuple, List
from kgpy.optics.zemax import ZOSAPI

__all__ = ['ILensDataEditor']


class ILensDataEditor:

    Editor = None                               # type: ZOSAPI.Editors.EditorType
    FirstColumn = None                          # type: ZOSAPI.Editors.LDE.SurfaceColumn
    LastColumn = None                           # type: ZOSAPI.Editors.LDE.SurfaceColumn
    MaxColumn = None                            # type: int
    MinColumn = None                            # type: int
    NumberOfNonSequentialSurfaces = None        # type: int
    NumberOfRows = None                         # type: int
    NumberOfSurfaces = None                     # type: int
    RowToSurfaceOffset = None                   # type: int
    StopSurface = None                          # type: int

    def AddRow(self) -> 'ZOSAPI.Editors.IEditorRow':
        pass

    def AddSurface(self) -> 'ZOSAPI.Editors.LDE.ILDERow':
        pass

    def CopySurfaces(self, fromSurfaceNumber: int, NumberOfSurfaces: int, toSurfaceNumber: int) -> int:
        pass

    def CopySurfaceFrom(self, fromEditor: 'ZOSAPI.Editors.LDE.ILensDataEditor', fromSurfaceNumber: int,
                        NumberOfSurfaces: int, toSurfaceNumber: int) -> int:
        pass

    def DeleteAllRows(self) -> int:
        pass

    def DeleteRowAt(self, pos: int) -> bool:
        pass

    def DeleteRowsAt(self, pos: int, numberOfRows: int) -> int:
        pass

    def FindLabel(self, label: int) -> Tuple[bool, int]:
        pass

    def GetApodization(self, px: float, py: float) -> float:
        pass

    def GetFirstOrderData(self) -> Tuple[float, float, float, float, float]:
        pass

    def GetGlass(self, surface: int) -> Tuple[bool, str, float, float, float]:
        pass

    def GetGlobalMatrix(self, surface: int) -> Tuple[bool, float, float, float, float, float, float, float, float, 
                                                     float, float, float, float]:
        pass

    def GetIndex(self, Surface: int, NumberOfWavelengths: int, indexAtWavelength: List[float]
                 ) -> Tuple[int, List[float]]:
        pass

    def GetLabel(self, Surface: int) -> Tuple[bool, int]:
        pass

    def GetPupil(self) -> Tuple['ZOSAPI.SystemData.ZemaxApertureType', float, float, float, float, float,
                                'ZOSAPI.Editors.LDE.PupilApodizationType', float]:
        pass

    def GetRowAt(self, pos: int) -> 'ZOSAPI.Editors.IEditorRow':
        pass

    def GetSag(self, Surface: int, X: float, Y: float) -> Tuple[bool, float, float]:
        pass

    def GetSurfaceAt(self, surfaceNumber: int) -> 'ZOSAPI.Editors.LDE.ILDERow':
        pass

    def GetTool_AddCoatingsToAllSurfaces(self) -> 'ZOSAPI.Editors.LDE.ILDETool_AddCoatingsToAllSurfaces':
        pass

    def HideEditor(self) -> None:
        pass

    def HideLDE(self):
        pass

    def InsertNewSurfaceAt(self, SurfaceNumber: int) -> 'ZOSAPI.Editors.LDE.ILDERow':
        pass

    def InsertRowAt(self, pos: int) -> 'ZOSAPI.Editors.IEditorRow':
        pass

    def RemoveSurfaceAt(self, SurfaceNumber: int) -> bool:
        pass

    def RemoveSurfacesAt(self, SurfaceNumber: int, numSurfaces: int) -> int:
        pass

    def RunTool_AddCoatingsToAllSurfaces(self, settings: 'ZOSAPI.Editors.LDE.ILDETool_AddCoatingsToAllSurfaces'):
        pass

    def RunTool_ConvertGlobalToLocalCoordinates(self, FirstSurface: int, LastSurface: int,
                                                order: 'ZOSAPI.Editors.LDE.ConversionOrder'
                                                ) -> 'ZOSAPI.Editors.LDE.CoordinateConversionResult':
        pass

    def RunTool_ConvertLocalToGlobalCoordinates(self, FirstSurface: int, LastSurface: int, referenceSurface: int
                                                ) -> 'ZOSAPI.Editors.LDE':
        pass

    def RunTool_ConvertSemiDiametersToCircularApertures(self):
        pass

    def RunTool_ConvertSemiDiametersToFloatingApertures(self):
        pass

    def RunTool_ConvertSemiDiametersToMaximumApertures(self):
        pass

    def RunTool_RemoveAllApertures(self):
        pass

    def RunTool_ReplaceVignettingWithApertures(self):
        pass

    def SetLabel(self, Surface: int, label: int) -> bool:
        pass

    def ShowEditor(self) -> bool:
        pass

    def ShowLDE(self) -> bool:
        pass