import win32com.client
from astropy import units as u

from kgpy.optics.zemax import ZOSAPI


def add_coordinate_break_surface_to_zemax_system(zemax_system: ZOSAPI.IOpticalSystem,
                                                 zemax_surface: ZOSAPI.Editors.LDE.ILDERow,
                                                 zemax_units: u.Unit,
                                                 configuration_index: int,
                                                 surface_index: int,
                                                 surface: 'optics.system.configuration.surface.CoordinateBreak'):

    n_params = 6

    if configuration_index == 0:
        for p in range(n_params):
            op = zemax_system.MCE.AddOperand()
            op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM)
            op.Param1 = surface_index
            op.Param2 = p + 1

    zemax_surface_settings = zemax_surface.GetSurfaceTypeSettings(
        ZOSAPI.Editors.LDE.SurfaceType.CoordinateBreak)

    zemax_surface.ChangeType(zemax_surface_settings)

    zemax_surface_data = win32com.client.CastTo(
        zemax_surface.SurfaceData,
        ZOSAPI.Editors.LDE.ISurfaceCoordinateBreak.__name__
    )  # type: ZOSAPI.Editors.LDE.ISurfaceCoordinateBreak



    if surface.tilt_decenter.translation_first:
        zemax_surface_data.Order = 0

    else:
        zemax_surface_data.Order = 1

    translation = surface.tilt_decenter.translation
    zemax_surface_data.Decenter_X = translation.x.to(zemax_units).value
    zemax_surface_data.Decenter_Y = translation.y.to(zemax_units).value

    rotation = surface.tilt_decenter.rotation

    zemax_surface_data.TiltAbout_X = rotation.x.to(u.deg).value
    zemax_surface_data.TiltAbout_Y = rotation.y.to(u.deg).value
    zemax_surface_data.TiltAbout_Z = rotation.z.to(u.deg).value