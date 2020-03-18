import typing as tp
from astropy import units as u

from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import util, surface

__all__ = ['add_to_zemax_system']


class CoordinateBreak(system.surface.CoordinateBreak, surface.Surface):

    @property
    def lde_row(self) -> ZOSAPI.Editors.LDE.ILDERow[ZOSAPI.Editors.LDE.ISurfaceCoordinateBreak]:
        return super().lde_row


def add_to_zemax_system(
        zemax_system: ZOSAPI.IOpticalSystem,
        surface: 'system.surface.CoordinateBreak',
        surface_index: int,
        configuration_shape: tp.Tuple[int],
        zemax_units: u.Unit,
):
    op_decenter_x = ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM
    op_decenter_y = ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM
    op_tilt_x = ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM
    op_tilt_y = ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM
    op_tilt_z = ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM
    op_order = ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM

    ind_decenter_x = 1
    ind_decenter_y = 2
    ind_tilt_x = 3
    ind_tilt_y = 4
    ind_tilt_z = 5
    ind_order = 6

    unit_decenter_x = zemax_units
    unit_decenter_y = zemax_units
    unit_tilt_x = u.deg
    unit_tilt_y = u.deg
    unit_tilt_z = u.deg
    unit_order = None

    zemax_surface = zemax_system.LDE.GetSurfaceAt(surface_index)
    zemax_surface.ChangeType(zemax_surface.GetSurfaceTypeSettings(ZOSAPI.Editors.LDE.SurfaceType.CoordinateBreak))

    util.set_float(zemax_system, surface.decenter.x, configuration_shape, op_decenter_x, unit_decenter_x,
                   surface_index, ind_decenter_x)
    util.set_float(zemax_system, surface.decenter.y, configuration_shape, op_decenter_y, unit_decenter_y,
                   surface_index, ind_decenter_y)
    util.set_float(zemax_system, surface.tilt.x, configuration_shape, op_tilt_x, unit_tilt_x, surface_index,
                   ind_tilt_x)
    util.set_float(zemax_system, surface.tilt.y, configuration_shape, op_tilt_y, unit_tilt_y, surface_index,
                   ind_tilt_y)
    util.set_float(zemax_system, surface.tilt.z, configuration_shape, op_tilt_z, unit_tilt_z, surface_index,
                   ind_tilt_z)
    util.set_float(zemax_system, float(surface.tilt_first.value), configuration_shape, op_order, unit_order,
                   surface_index, ind_order)
