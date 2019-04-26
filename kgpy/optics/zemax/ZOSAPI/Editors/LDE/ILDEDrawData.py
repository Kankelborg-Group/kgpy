
from kgpy.optics.zemax import ZOSAPI

__all__ = []


class ILDEDrawData:

    DoNotDrawEdgesFromThisSurface = None    # type: bool
    DoNotDrawThisSurface = None             # type: bool
    DrawEdgesAs = None                      # type: ZOSAPI.Editors.LDE.SurfaceEdgeDraw
    DrawLocalAxis = None                    # type: bool
    HasMirrorSettings = None                # type: bool
    HideRaysToThisSurface = None            # type: bool
    MirrorSubstrate = None                  # type: ZOSAPI.Editors.LDE.SubstrateType
    MirrorThickness = None                  # type: float
    SkipRaysToThisSurface = None            # type: bool
