from astropy import units as u

from kgpy.optics.zemax import ZOSAPI


def add_aperture_to_zemax_surface(zemax_system: ZOSAPI.IOpticalSystem,
                                  zemax_surface: ZOSAPI.Editors.LDE.ILDERow,
                                  zemax_units: u.Unit,
                                  configuration_index: int,
                                  surface_index: int,
                                  aperture: 'optics.system.configuration.surface.Aperture'):
    aperture_op_types = [
        ZOSAPI.Editors.MCE.MultiConfigOperandType.APTP,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.APDX,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.APDY,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.APMN,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.APMX
    ]

    if isinstance(aperture, kgpy.optics.system.surface.aperture.Rectangular):

        if configuration_index == 0:

            for op_type in aperture_op_types:
                op = zemax_system.MCE.AddOperand()
                op.ChangeType(op_type)
                op.Param1 = surface_index

        if aperture.is_obscuration:
            zemax_aperture_type = ZOSAPI.Editors.LDE.SurfaceApertureTypes.RectangularObscuration
        else:
            zemax_aperture_type = ZOSAPI.Editors.LDE.SurfaceApertureTypes.RectangularAperture

        zemax_aperture_settings = zemax_surface.ApertureData.CreateApertureTypeSettings(
            zemax_aperture_type)._S_RectangularAperture  # type: ZOSAPI.Editors.LDE.ISurfaceApertureRectangular

        zemax_aperture_settings.XHalfWidth = aperture.half_width_x.to(zemax_units).value
        zemax_aperture_settings.YHalfWidth = aperture.half_width_y.to(zemax_units).value
        zemax_aperture_settings.ApertureXDecenter = aperture.decenter_x.to(zemax_units).value
        zemax_aperture_settings.ApertureYDecenter = aperture.decenter_y.to(zemax_units).value

        zemax_surface.ApertureData.ChangeApertureTypeSettings(zemax_aperture_settings)