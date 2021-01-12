
from kgpy.optics.zemax import ZOSAPI

__all__ = ['ISDApertureData']


class ISDApertureData:

    AFocalImageSpace = None             # type: bool
    ApertureType = None                 # type: ZOSAPI.SystemData.ZemaxApertureType
    ApertureValue = None                # type: float
    ApodizationFactor = None            # type: float
    ApodizationFactorIsUsed = None      # type: bool
    ApodizationType = None              # type: ZOSAPI.SystemData.ZemaxApodizationType
    CheckGRINApertures = None           # type: bool
    FastSemiDiameters = None            # type: bool
    GCRS = None                         # type: ZOSAPI.SystemData.ISurfaceSelection
    IterateSolvesWhenUpdating = None    # type: bool
    SemiDiameterMargin = None           # type: float
    SemiDiameterMarginPct = None        # type: float
    TelecentricObjectSpace = None       # type: bool
