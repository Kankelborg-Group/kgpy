
from kgpy.optics.zemax import ZOSAPI

__all__ = ['ISurface', 'ISurfaceStandard', 'ISurfaceCoordinateBreak', 'ISurfaceNthZernike', 'ISurfaceToroidal',
           'ISurfaceDiffractionGrating', 'ISurfaceEllipticalGrating1']


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
    LinesPerMicroMeter = None           # type: float
    LinesPerMicroMeterCell = None       # type: ZOSAPI.Editors.IEditorCell


class ISurfaceEllipticalGrating1(ISurface):

    A = None                            # type: float
    aCell = None                        # type: ZOSAPI.Editors.IEditorCell
    Alpha = None                        # type: float
    AlphaCell = None                    # type: ZOSAPI.Editors.IEditorCell
    B = None                            # type: float
    bCell = None                        # type: ZOSAPI.Editors.IEditorCell
    Beta = None                         # type: float
    BetaCell = None                     # type: ZOSAPI.Editors.IEditorCell
    c = None                            # type: float
    cCell = None                        # type: ZOSAPI.Editors.IEditorCell
    Delta = None                        # type: float
    DeltaCell = None                    # type: ZOSAPI.Editors.IEditorCell
    DiffractionOrder = None             # type: float
    DiffractionOrderCell = None         # type: ZOSAPI.Editors.IEditorCell
    Epsilon = None                      # type: float
    EpsilonCell = None                  # type: ZOSAPI.Editors.IEditorCell
    Gamma = None                        # type: float
    GammaCell = None                    # type: ZOSAPI.Editors.IEditorCell
    LinesPerMicroMeter = None           # type: float
    LinesPerMicroMeterCell = None       # type: ZOSAPI.Editors.IEditorCell
    NormRadius = None                   # type: float
    NormRadiusCell = None               # type: ZOSAPI.Editors.IEditorCell
    NumberOfTerms = None                # type: int
    NumberOfTermsCell = None            # type: ZOSAPI.Editors.IEditorCell

