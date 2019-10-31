import typing as tp
from astropy import units as u

from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system.surface.aperture import add_to_zemax_surface

from .. import util


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

    standard_surface_op_types = [
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CRVT,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.GLSS,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CONN,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CBDX,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CBDY,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CBTX,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CBTY,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CBTZ,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CBOR,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CADX,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CADY,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CATX,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CATY,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CATZ,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CAOR,
    ]

    if configuration_index == 0:

        for op_type in standard_surface_op_types:
            op = zemax_system.MCE.AddOperand()
            op.ChangeType(op_type)
            op.Param1 = surface_index

    zemax_surface.Radius = surface.radius.to(zemax_units).value
    zemax_surface.Material = str(surface.material)
    zemax_surface.Conic = float(surface.conic)

    if surface.pre_tilt_decenter.translation_first:
        zemax_surface.TiltDecenterData.BeforeSurfaceOrder = 0

    else:
        zemax_surface.TiltDecenterData.BeforeSurfaceOrder = 1

    translation = surface.pre_tilt_decenter.translation
    zemax_surface.TiltDecenterData.BeforeSurfaceDecenterX = translation.x.to(zemax_units).value
    zemax_surface.TiltDecenterData.BeforeSurfaceDecenterY = translation.y.to(zemax_units).value

    rotation = surface.pre_tilt_decenter.rotation

    zemax_surface.TiltDecenterData.BeforeSurfaceTiltX = rotation.x.to(u.deg).value
    zemax_surface.TiltDecenterData.BeforeSurfaceTiltY = rotation.y.to(u.deg).value
    zemax_surface.TiltDecenterData.BeforeSurfaceTiltZ = rotation.z.to(u.deg).value

    if surface.post_tilt_decenter.translation_first:
        zemax_surface.TiltDecenterData.AfterSurfaceOrder = 0

    else:
        zemax_surface.TiltDecenterData.AfterSurfaceOrder = 1

    translation = surface.post_tilt_decenter.translation
    zemax_surface.TiltDecenterData.AfterSurfaceDecenterX = translation.x.to(zemax_units).value
    zemax_surface.TiltDecenterData.AfterSurfaceDecenterY = translation.y.to(zemax_units).value

    rotation = surface.pre_tilt_decenter.rotation

    zemax_surface.TiltDecenterData.AfterSurfaceTiltX = rotation.x.to(u.deg).value
    zemax_surface.TiltDecenterData.AfterSurfaceTiltY = rotation.y.to(u.deg).value
    zemax_surface.TiltDecenterData.AfterSurfaceTiltZ = rotation.z.to(u.deg).value

    add_to_zemax_surface(zemax_system, zemax_surface, zemax_units, configuration_index, surface_index,
                         surface.aperture)
