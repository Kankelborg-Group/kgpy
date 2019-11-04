import typing as tp
from astropy import units as u

from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import util

__all__ = ['add_to_zemax_system']


def add_to_zemax_system(
        zemax_system: ZOSAPI.IOpticalSystem,
        surface: 'system.surface.EllipticalGrating1',
        surface_index: int,
        configuration_shape: tp.Tuple[int],
        zemax_units: u.Unit,
):
    op_a = ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM
    op_b = ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM
    op_c = ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM
    op_alpha = ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM
    op_beta = ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM

    ind_a = 3
    ind_b = 4
    ind_c = 5
    ind_alpha = 6
    ind_beta = 7

    unit_a = u.dimensionless_unscaled
    unit_b = u.dimensionless_unscaled
    unit_c = zemax_units
    unit_alpha = u.dimensionless_unscaled
    unit_beta = u.dimensionless_unscaled

    zemax_surface = zemax_system.LDE.GetSurfaceAt(surface_index)
    zemax_surface.ChangeType(zemax_surface.GetSurfaceTypeSettings(ZOSAPI.Editors.LDE.SurfaceType.EllipticalGrating1))

    util.set_float(zemax_system, surface.a, configuration_shape, op_a, unit_a, surface_index, ind_a)
    util.set_float(zemax_system, surface.b, configuration_shape, op_b, unit_b, surface_index, ind_b)
    util.set_float(zemax_system, surface.c, configuration_shape, op_c, unit_c, surface_index, ind_c)
    util.set_float(zemax_system, surface.alpha, configuration_shape, op_alpha, unit_alpha, surface_index, ind_alpha)
    util.set_float(zemax_system, surface.beta, configuration_shape, op_beta, unit_beta, surface_index, ind_beta)

