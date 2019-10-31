import win32com.client
from astropy import units as u

from kgpy.optics.zemax import ZOSAPI


def add_diffraction_grating_to_zemax_system(zemax_system: ZOSAPI.IOpticalSystem,
                                  zemax_surface: ZOSAPI.Editors.LDE.ILDERow,
                                  configuration_index: int,
                                  surface_index: int,
                                  surface: 'optics.system.configuration.surface.DiffractionGrating'):

    n_params = 2

    if configuration_index == 0:

        for p in range(n_params):
            op = zemax_system.MCE.AddOperand()
            op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM)
            op.Param1 = surface_index
            op.Param2 = p + 1

    zemax_surface_settings = zemax_surface.GetSurfaceTypeSettings(
        ZOSAPI.Editors.LDE.SurfaceType.DiffractionGrating)

    zemax_surface.ChangeType(zemax_surface_settings)

    zemax_surface_data = win32com.client.CastTo(
        zemax_surface.SurfaceData,
        ZOSAPI.Editors.LDE.ISurfaceDiffractionGrating.__name__
    )  # type: ZOSAPI.Editors.LDE.ISurfaceDiffractionGrating

    zemax_surface_data.DiffractionOrder = surface.diffraction_order
    zemax_surface_data.LinesPerMicroMeter = surface.groove_frequency.to(1 / u.um).value