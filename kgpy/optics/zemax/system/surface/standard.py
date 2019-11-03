import typing as tp
from astropy import units as u

from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI

from . import diffraction_grating, toroidal, aperture
from .. import util

__all__ = ['add_to_zemax_system']


def add_to_zemax_system(
        zemax_system: ZOSAPI.IOpticalSystem,
        surface: 'system.surface.Standard',
        surface_index: int,
        configuration_shape: tp.Tuple[int],
        zemax_units: u.Unit,
):

    op_radius = ZOSAPI.Editors.MCE.MultiConfigOperandType.CRVT
    op_material = ZOSAPI.Editors.MCE.MultiConfigOperandType.GLSS
    op_conic = ZOSAPI.Editors.MCE.MultiConfigOperandType.CONN
    op_decenter_before_x = ZOSAPI.Editors.MCE.MultiConfigOperandType.CBDX
    op_decenter_before_y = ZOSAPI.Editors.MCE.MultiConfigOperandType.CBDY
    op_tilt_before_x = ZOSAPI.Editors.MCE.MultiConfigOperandType.CBTX
    op_tilt_before_y = ZOSAPI.Editors.MCE.MultiConfigOperandType.CBTY
    op_tilt_before_z = ZOSAPI.Editors.MCE.MultiConfigOperandType.CBTZ
    op_order_before = ZOSAPI.Editors.MCE.MultiConfigOperandType.CBOR,
    op_decenter_after_x = ZOSAPI.Editors.MCE.MultiConfigOperandType.CADX,
    op_decenter_after_y = ZOSAPI.Editors.MCE.MultiConfigOperandType.CADY,
    op_tilt_after_x = ZOSAPI.Editors.MCE.MultiConfigOperandType.CATX,
    op_tilt_after_y = ZOSAPI.Editors.MCE.MultiConfigOperandType.CATY,
    op_tilt_after_z = ZOSAPI.Editors.MCE.MultiConfigOperandType.CATZ,
    op_order_after = ZOSAPI.Editors.MCE.MultiConfigOperandType.CAOR,

    unit_radius = zemax_units
    unit_conic = u.dimensionless_unscaled
    unit_decenter_before_x = zemax_units
    unit_decenter_before_y = zemax_units
    unit_tilt_before_x = u.deg
    unit_tilt_before_y = u.deg
    unit_tilt_before_z = u.deg
    unit_order_before = None
    unit_decenter_after_x = zemax_units
    unit_decenter_after_y = zemax_units
    unit_tilt_after_x = u.deg
    unit_tilt_after_y = u.deg
    unit_tilt_after_z = u.deg
    unit_order_after = None

    util.set_float(zemax_system, surface.radius, configuration_shape, op_radius, unit_radius, surface_index)
    util.set_str(zemax_system, surface.material.name, configuration_shape, op_material, surface_index)
    util.set_float(zemax_system, surface.conic, configuration_shape, op_conic, unit_conic, surface_index)

    util.set_float(zemax_system, surface.decenter_before[..., 0], configuration_shape, op_decenter_before_x,
                   unit_decenter_before_x, surface_index)
    util.set_float(zemax_system, surface.decenter_before[..., 1], configuration_shape, op_decenter_before_y,
                   unit_decenter_before_y, surface_index)
    util.set_float(zemax_system, surface.tilt_before[..., 0], configuration_shape, op_tilt_before_x,
                   unit_tilt_before_x, surface_index)
    util.set_float(zemax_system, surface.tilt_before[..., 1], configuration_shape, op_tilt_before_y,
                   unit_tilt_before_y, surface_index)
    util.set_float(zemax_system, surface.tilt_before[..., 2], configuration_shape, op_tilt_before_z,
                   unit_tilt_before_z, surface_index)
    util.set_float(zemax_system, float(surface.tilt_first), configuration_shape, op_order_before, unit_order_before,
                   surface_index)

    util.set_float(zemax_system, surface.decenter_after[..., 0], configuration_shape, op_decenter_after_x,
                   unit_decenter_after_x, surface_index)
    util.set_float(zemax_system, surface.decenter_after[..., 1], configuration_shape, op_decenter_after_y,
                   unit_decenter_after_y, surface_index)
    util.set_float(zemax_system, surface.tilt_after[..., 0], configuration_shape, op_tilt_after_x,
                   unit_tilt_after_x, surface_index)
    util.set_float(zemax_system, surface.tilt_after[..., 1], configuration_shape, op_tilt_after_y,
                   unit_tilt_after_y, surface_index)
    util.set_float(zemax_system, surface.tilt_after[..., 2], configuration_shape, op_tilt_after_z,
                   unit_tilt_after_z, surface_index)
    util.set_float(zemax_system, float(surface.tilt_first), configuration_shape, op_order_after, unit_order_after,
                   surface_index)

    if isinstance(surface, system.surface.DiffractionGrating):
        diffraction_grating.add_to_zemax_system(zemax_system, surface, surface_index, configuration_shape, zemax_units)

    elif isinstance(surface, system.surface.Toroidal):
        toroidal.add_to_zemax_system(zemax_system, surface, surface_index, configuration_shape, zemax_units)

    aperture.add_to_zemax_surface(zemax_system, surface.aperture, surface_index, configuration_shape, zemax_units)
