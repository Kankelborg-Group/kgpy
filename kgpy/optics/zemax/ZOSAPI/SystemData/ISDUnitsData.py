
from kgpy.optics.zemax import ZOSAPI

__all__ = []


class ISDUnitsData:

    AfocalModeUnits = None      # type: ZOSAPI.SystemData.ZemaxAfocalModeUnits
    AnalysisUnitPrefix = None   # type: ZOSAPI.SystemData.ZemaxUnitPrefix
    AnalysisUnits = None        # type: ZOSAPI.SystemData.ZemaxAnalysisUnits
    LensUnits = None            # type: ZOSAPI.SystemData.ZemaxSystemUnits
    MTFUnits = None             # type: ZOSAPI.SystemData.ZemaxMTFUnits
    SourceUnitPrefix = None     # type: ZOSAPI.SystemData.ZemaxUnitPrefix
    SourceUnits = None          # type: ZOSAPI.SystemData.ZemaxSourceUnits
