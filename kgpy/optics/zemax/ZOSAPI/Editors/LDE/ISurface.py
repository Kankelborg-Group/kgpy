
from kgpy.optics.zemax import ZOSAPI

__all__ = ['ISurface', 'ISurfaceStandard', 'ISurfaceCoordinateBreak']


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


