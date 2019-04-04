
from kgpy.optics.zemax import ZOSAPI

__all__ = ['ISurface', 'ISurfaceStandard', 'ISurfaceCoordinateBreak', 'ISurfaceNthZernike', 'ISurfaceToroidal',
           'ISurfaceDiffractionGrating']


class ISurface:

    IsValid = None      # type: bool
    Row = None          # type: ZOSAPI.Editors.LDE.ILDERow
    Type = None         # type: ZOSAPI.Editors.LDE.SurfaceType


class ISurfaceStandard(ISurface):

    pass


class ISurfaceCoordinateBreak(ISurface):

    Decenter_X = None           # type: float
    Decenter_X_Cell = None      # type: ZOSAPI.Editors.IEditorCell
    Decenter_Y = None           # type: float
    Decenter_Y_Cell = None      # type: ZOSAPI.Editors.IEditorCell
    Order = None                # type: int
    OrderCell = None            # type: ZOSAPI.Editors.IEditorCell
    TiltAbout_X = None          # type: float
    TiltAbout_X_Cell = None     # type: ZOSAPI.Editors.IEditorCell
    TiltAbout_Y = None          # type: float
    TiltAbout_Y_Cell = None     # type: ZOSAPI.Editors.IEditorCell
    TiltAbout_Z = None          # type: float
    TiltAbout_Z_Cell = None     # type: ZOSAPI.Editors.IEditorCell


class ISurfaceNthZernike(ISurface):

    NormRadius = None           # type: float
    NormRadiusCell = None       # type: ZOSAPI.Editors.IEditorCell
    NumberOfTerms = None        # type: float
    NumberOfTermsCell = None    # type: ZOSAPI.Editors.IEditorCell

    def GetNthZernikeCoefficient(self, n: int) -> float:
        pass

    def NthZernikeCoefficient(self, n: int) -> 'ZOSAPI.Editors.IEditorCell':
        pass

    def SetNthZernikeCoefficient(self, n: int, value: float) -> None:
        pass


class ISurfaceToroidal(ISurfaceNthZernike):

    Extrapolate = None              # type: int
    ExtrapolateCell = None          # type: ZOSAPI.Editors.IEditorCell
    RadiusOfRotation = None         # type: float
    RadiusOfRotationCell = None     # type: ZOSAPI.Editors.IEditorCell


class ISurfaceDiffractionGrating(ISurface):

    DiffractionOrder = None             # type: int
    DiffractionOrderCell = None         # type: ZOSAPI.Editors.IEditorCell
    LinesPerMicrometer = None           # type: float
    LinesPerMicrometerCell = None       # type: ZOSAPI.Editors.IEditorCell

