import dataclasses
import typing as typ
from astropy import units as u

from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI

from ... import util, configuration
from .. import diffraction_grating, toroidal, aperture, material
from .. import Surface
from . import coordinate

__all__ = ['Standard', 'add_to_zemax_system']


class InstanceVarBase:

    _radius_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.CRVT
        ),
        init=None,
        repr=None,
    )
    _conic_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.CONN
        ),
        init=None,
        repr=None,
    )



class Standard(system.surface.Standard, Surface):
    pass


def add_to_zemax_system(
        zemax_system: ZOSAPI.IOpticalSystem,
        surface: 'system.surface.Standard',
        surface_index: int,
        configuration_shape: typ.Tuple[int],
        zemax_units: u.Unit,
):

    op_radius = ZOSAPI.Editors.MCE.MultiConfigOperandType.CRVT
    op_conic = ZOSAPI.Editors.MCE.MultiConfigOperandType.CONN
    op_decenter_before_x = ZOSAPI.Editors.MCE.MultiConfigOperandType.CBDX
    op_decenter_before_y = ZOSAPI.Editors.MCE.MultiConfigOperandType.CBDY
    op_tilt_before_x = ZOSAPI.Editors.MCE.MultiConfigOperandType.CBTX
    op_tilt_before_y = ZOSAPI.Editors.MCE.MultiConfigOperandType.CBTY
    op_tilt_before_z = ZOSAPI.Editors.MCE.MultiConfigOperandType.CBTZ
    op_order_before = ZOSAPI.Editors.MCE.MultiConfigOperandType.CBOR
    op_decenter_after_x = ZOSAPI.Editors.MCE.MultiConfigOperandType.CADX
    op_decenter_after_y = ZOSAPI.Editors.MCE.MultiConfigOperandType.CADY
    op_tilt_after_x = ZOSAPI.Editors.MCE.MultiConfigOperandType.CATX
    op_tilt_after_y = ZOSAPI.Editors.MCE.MultiConfigOperandType.CATY
    op_tilt_after_z = ZOSAPI.Editors.MCE.MultiConfigOperandType.CATZ
    op_order_after = ZOSAPI.Editors.MCE.MultiConfigOperandType.CAOR

    unit_radius = 1 / zemax_units
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

    util.set_float(zemax_system, 1 / surface.radius, configuration_shape, op_radius, unit_radius, surface_index)

    util.set_float(zemax_system, surface.conic, configuration_shape, op_conic, unit_conic, surface_index)

    material.add_to_zemax_surface(zemax_system, surface.material, surface_index, configuration_shape, zemax_units)

    aperture.add_to_zemax_surface(zemax_system, surface.aperture, surface_index, configuration_shape, zemax_units)

    util.set_float(zemax_system, surface.transform_before.decenter.x, configuration_shape, op_decenter_before_x,
                   unit_decenter_before_x, surface_index)
    util.set_float(zemax_system, surface.transform_before.decenter.y, configuration_shape, op_decenter_before_y,
                   unit_decenter_before_y, surface_index)
    util.set_float(zemax_system, surface.transform_before.tilt.x, configuration_shape, op_tilt_before_x,
                   unit_tilt_before_x, surface_index)
    util.set_float(zemax_system, surface.transform_before.tilt.y, configuration_shape, op_tilt_before_y,
                   unit_tilt_before_y, surface_index)
    util.set_float(zemax_system, surface.transform_before.tilt.z, configuration_shape, op_tilt_before_z,
                   unit_tilt_before_z, surface_index)
    util.set_float(zemax_system, float(surface.transform_before.tilt_first.value), configuration_shape, op_order_before,
                   unit_order_before, surface_index)

    util.set_float(zemax_system, surface.transform_after.decenter.x, configuration_shape, op_decenter_after_x,
                   unit_decenter_after_x, surface_index)
    util.set_float(zemax_system, surface.transform_after.decenter.y, configuration_shape, op_decenter_after_y,
                   unit_decenter_after_y, surface_index)
    util.set_float(zemax_system, surface.transform_after.tilt.x, configuration_shape, op_tilt_after_x,
                   unit_tilt_after_x, surface_index)
    util.set_float(zemax_system, surface.transform_after.tilt.y, configuration_shape, op_tilt_after_y,
                   unit_tilt_after_y, surface_index)
    util.set_float(zemax_system, surface.transform_after.tilt.z, configuration_shape, op_tilt_after_z,
                   unit_tilt_after_z, surface_index)
    util.set_float(zemax_system, float(not surface.transform_after.tilt_first.value), configuration_shape,
                   op_order_after, unit_order_after, surface_index)

    if isinstance(surface, system.surface.DiffractionGrating):
        diffraction_grating.add_to_zemax_system(zemax_system, surface, surface_index, configuration_shape, zemax_units)

    elif isinstance(surface, system.surface.Toroidal):
        toroidal.add_to_zemax_system(zemax_system, surface, surface_index, configuration_shape, zemax_units)






