import typing as tp
import astropy.units as u

from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI

from . import standard, coordinate_break
from .. import util

__all__ = ['add_surfaces_to_zemax_system']


def add_surfaces_to_zemax_system(
        zemax_system: ZOSAPI.IOpticalSystem,
        surfaces: 'tp.Iterable[system.Surface]',
        configuration_shape: tp.Tuple[int],
        zemax_units: u.Unit,

):

    op_comment = ZOSAPI.Editors.MCE.MultiConfigOperandType.MCOM
    op_thickness = ZOSAPI.Editors.MCE.MultiConfigOperandType.THIC
    op_is_visible = ZOSAPI.Editors.MCE.MultiConfigOperandType.SDRW

    unit_thickness = zemax_units

    surfaces = list(surfaces)
    num_surfaces = len(surfaces)
    while zemax_system.LDE.NumberOfSurfaces < num_surfaces + 1:
        zemax_system.LDE.AddSurface()
    
    for s in range(num_surfaces):
        
        surface_index = s + 1
        surface = surfaces[s]
        
        util.set_str(zemax_system, surface.name.__str__(), configuration_shape, op_comment, surface_index)
        util.set_float(zemax_system, surface.thickness, configuration_shape, op_thickness, unit_thickness,
                       surface_index)
        util.set_int(zemax_system, not surface.is_visible, configuration_shape, op_is_visible, surface_index)
        
        if isinstance(surface, system.surface.Standard):
            standard.add_to_zemax_system(zemax_system, surface, surface_index, configuration_shape, zemax_units)

        elif isinstance(surface, system.surface.CoordinateBreak):
            coordinate_break.add_to_zemax_system(zemax_system, surface, surface_index, configuration_shape, zemax_units)
