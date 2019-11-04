import typing as tp
import pathlib
from astropy import units as u

from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import util

from . import mirror

__all__ = ['add_to_zemax_surface']


def add_to_zemax_surface(
        zemax_system: ZOSAPI.IOpticalSystem,
        material: 'system.surface.Material',
        surface_index: int,
        configuration_shape: tp.Tuple[int],
        zemax_units: u.Unit,
):
    op_material = ZOSAPI.Editors.MCE.MultiConfigOperandType.GLSS
    
    if material is None:
        util.set_str(zemax_system, '', configuration_shape, op_material, surface_index)
        return

    util.set_str(zemax_system, material.name, configuration_shape, op_material, surface_index)
    
    if isinstance(material, system.surface.material.Mirror):
        mirror.add_to_zemax_surface(zemax_system, material, surface_index, zemax_units)
