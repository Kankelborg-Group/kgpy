
from kgpy.optics.zemax import ZOSAPI

__all__ = ['ILDEApertureData']


class ILDEApertureData:

    CurrentType = None                                          # type: ZOSAPI.Editors.LDE.SurfaceApertureTypes
    CurrentTypeSettings = None                                  # type: ZOSAPI.Editors.LDE.ISurfaceApertureType
    DisableClearSemiDiameterMarginsForThisSurface = None        # type: bool
    IsPickedUp = None                                           # type: bool
    PickupFrom = None                                           # type: int

    def ChangeApertureTypeSettings(self, settings: 'ZOSAPI.Editors.LDE.ISurfaceApertureType') -> bool:
        pass

    def CreateApertureTypeSettings(self, type: 'ZOSAPI.Editors.LDE.SurfaceApertureTypes'
                                   ) -> 'ZOSAPI.Editors.LDE.ISurfaceApertureType':
        pass

    def SetPickupNone(self) -> None:
        pass
