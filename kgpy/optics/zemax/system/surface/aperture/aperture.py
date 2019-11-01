import typing as tp
from astropy import units as u

from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import util

from . import rectangular, circular, spider

__all__ = ['add_to_zemax_surface']


def add_to_zemax_surface(
        zemax_system: ZOSAPI.IOpticalSystem,
        aperture: 'system.surface.Aperture',
        surface_index: int,
        configuration_shape: tp.Tuple[int],
        zemax_units: u.Unit,
):
    if aperture is None:
        type_ind = ZOSAPI.Editors.LDE.SurfaceApertureTypes.none
        op_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.APTP
        util.set_int(zemax_system, type_ind, configuration_shape, op_type, surface_index)
        return

    op_decenter_x = ZOSAPI.Editors.MCE.MultiConfigOperandType.APDX
    op_decenter_y = ZOSAPI.Editors.MCE.MultiConfigOperandType.APDY

    unit_decenter_x = zemax_units
    unit_decenter_y = zemax_units

    util.set_float(zemax_system, aperture.decenter_x, configuration_shape, op_decenter_x, unit_decenter_x,
                   surface_index)
    util.set_float(zemax_system, aperture.decenter_y, configuration_shape, op_decenter_y, unit_decenter_y,
                   surface_index)

    if isinstance(aperture, system.surface.aperture.Rectangular):
        rectangular.add_to_zemax_surface(zemax_system, aperture, surface_index, configuration_shape, zemax_units)

    elif isinstance(aperture, system.surface.aperture.Circular):
        circular.add_to_zemax_surface(zemax_system, aperture, surface_index, configuration_shape, zemax_units)

    elif isinstance(aperture, system.surface.aperture.Spider):
        spider.add_to_zemax_surface(zemax_system, aperture, surface_index, configuration_shape, zemax_units)


