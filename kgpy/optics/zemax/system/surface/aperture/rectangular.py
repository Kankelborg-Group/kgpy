import typing as tp
from astropy import units as u

from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import util

__all__ = ['add_to_zemax_surface']


def add_to_zemax_surface(
        zemax_system: ZOSAPI.IOpticalSystem,
        aperture: 'system.surface.aperture.Rectangular',
        surface_index: int,
        configuration_shape: tp.Tuple[int],
        zemax_units: u.Unit,
):

    if aperture.is_obscuration:
        type_ind = ZOSAPI.Editors.LDE.SurfaceApertureTypes.RectangularObscuration
    else:
        type_ind = ZOSAPI.Editors.LDE.SurfaceApertureTypes.RectangularAperture

    op_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.APTP
    op_half_width_x = ZOSAPI.Editors.MCE.MultiConfigOperandType.APMN
    op_half_width_y = ZOSAPI.Editors.MCE.MultiConfigOperandType.APMX

    unit_half_width_x = zemax_units
    unit_half_width_y = zemax_units

    util.set_int(zemax_system, type_ind, configuration_shape, op_type, surface_index)
    util.set_float(zemax_system, aperture.half_width_x, configuration_shape, op_half_width_x, unit_half_width_x,
                   surface_index)
    util.set_float(zemax_system, aperture.half_width_y, configuration_shape, op_half_width_y, unit_half_width_y,
                   surface_index)
