import typing as tp
from astropy import units as u

from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import util

__all__ = ['add_to_zemax_system']


def add_to_zemax_system(
        zemax_system: ZOSAPI.IOpticalSystem,
        surface: 'system.surface.Toroidal',
        surface_index: int,
        configuration_shape: tp.Tuple[int],
        zemax_units: u.Unit,
):

    op_radius_of_rotation = ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM

    ind_radius_of_rotation = 1

    unit_radius_of_rotation = zemax_units
    
    zemax_surface = zemax_system.LDE.GetSurfaceAt(surface_index)
    zemax_surface.ChangeType(zemax_surface.GetSurfaceTypeSettings(ZOSAPI.Editors.LDE.SurfaceType.Toroidal))

    util.set_float(zemax_system, surface.radius_of_rotation, configuration_shape, op_radius_of_rotation,
                   unit_radius_of_rotation, surface_index, ind_radius_of_rotation)


