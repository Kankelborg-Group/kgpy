import typing as tp
from astropy import units as u

from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI

from kgpy.optics.zemax.system import util

__all__ = ['add_to_zemax_surface']


def add_to_zemax_surface(
        zemax_system: ZOSAPI.IOpticalSystem,
        aperture: 'system.surface.aperture.Spider',
        surface_index: int,
        configuration_shape: tp.Tuple[int],
        zemax_units: u.Unit,
):

    type_ind = ZOSAPI.Editors.LDE.SurfaceApertureTypes.Spider

    op_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.APTP
    op_arm_half_width = ZOSAPI.Editors.MCE.MultiConfigOperandType.APMN
    op_num_arms = ZOSAPI.Editors.MCE.MultiConfigOperandType.APMX

    unit_arm_half_width = zemax_units
    unit_num_arms = None

    util.set_int(zemax_system, type_ind, configuration_shape, op_type, surface_index)
    util.set_float(zemax_system, 2 * aperture.arm_half_width, configuration_shape, op_arm_half_width, unit_arm_half_width,
                   surface_index)
    util.set_float(zemax_system, aperture.num_arms, configuration_shape, op_num_arms, unit_num_arms,
                   surface_index)
