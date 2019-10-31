
import typing as tp
import astropy.units as u

from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system.surface.coordinate_break import add_coordinate_break_surface_to_zemax_system
from kgpy.optics.zemax.system.surface.diffraction_grating import add_diffraction_grating_to_zemax_system
from kgpy.optics.zemax.system.surface.standard import add_standard_surface_to_zemax_system
from kgpy.optics.zemax.system.surface.toroidal import add_toroidal_surface_to_zemax_system


def add_surfaces_to_zemax_system(zemax_system: ZOSAPI.IOpticalSystem,
                                 zemax_units: u.Unit,
                                 configuration_index: int,
                                 surfaces: 'tp.List[optics.system.configuration.Surface]'):
    surface_op_types = [
        ZOSAPI.Editors.MCE.MultiConfigOperandType.MCOM,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.THIC,
    ]

    zemax_stop_surface = None

    for surface_index, surface in enumerate(surfaces):

        if configuration_index == 0:

            if surface_index >= zemax_system.LDE.NumberOfSurfaces - 1 <= surface_index < len(surfaces) - 1:
                zemax_system.LDE.AddSurface()

            for op_type in surface_op_types:
                op = zemax_system.MCE.AddOperand()
                op.ChangeType(op_type)
                op.Param1 = surface_index


        zemax_surface = zemax_system.LDE.GetSurfaceAt(surface_index)

        zemax_surface.Comment = surface.name
        # zemax_surface.IsStop = surface.is_stop
        zemax_surface.Thickness = surface.thickness.to(zemax_units).value

        if surface.is_stop:
            zemax_surface.IsStop = surface.is_stop

        if isinstance(surface, kgpy.optics.system.surface.Standard):

            add_standard_surface_to_zemax_system(zemax_system, zemax_surface, zemax_units, configuration_index,
                                                 surface_index, surface)

            if isinstance(surface, kgpy.optics.system.surface.DiffractionGrating):

                add_diffraction_grating_to_zemax_system(zemax_system, zemax_surface, configuration_index, surface_index,
                                                        surface)

            if isinstance(surface, kgpy.optics.system.surface.Toroidal):

                add_toroidal_surface_to_zemax_system(zemax_system, zemax_surface, zemax_units, configuration_index,
                                                     surface_index, surface)

        if isinstance(surface, kgpy.optics.system.surface.CoordinateBreak):

            add_coordinate_break_surface_to_zemax_system(zemax_system, zemax_surface, zemax_units, configuration_index,
                                                         surface_index, surface)






