
from kgpy.optics.zemax import ZOSAPI

__all__ = ['ISystemData']


class ISystemData:

    Advanced = None                 # type: ZOSAPI.SystemData.ISDAdvancedData
    Aperture = None                 # type: ZOSAPI.SystemData.ISDApertureData
    Environment = None              # type: ZOSAPI.SystemData.ISDEnvironmentData
    Fields = None                   # type: ZOSAPI.SystemData.IFields
    Files = None                    # type: ZOSAPI.SystemData.ISDFiles
    MaterialCatalogs = None         # type: ZOSAPI.SystemData.ISDMaterialCatalogData
    NamedFiltersData = None         # type: ZOSAPI.SystemData.ISDNamedFilters
    NonSequentialData = None        # type: ZOSAPI.SystemData.ISDNonSeqData
    Polarization = None             # type: ZOSAPI.SystemData.ISDPolarizationData
    RayAiming = None                # type: ZOSAPI.SystemData.ISDRayAimingData
    TitleNotes = None               # type: ZOSAPI.SystemData.ISDTitleNotes
    Units = None                    # type: ZOSAPI.SystemData.ISDUnitsData
    Wavelengths = None              # type: ZOSAPI.SystemData.IWavelengths
