import typing as tp
from astropy import units as u

from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import util

from . import elliptical_grating

__all__ = ['add_to_zemax_system']


def add_to_zemax_system(
        zemax_system: ZOSAPI.IOpticalSystem,
        surface: 'system.surface.DiffractionGrating',
        surface_index: int,
        configuration_shape: tp.Tuple[int],
        zemax_units: u.Unit,
):
    op_groove_frequency = ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM
    op_diffraction_order = ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM

    ind_groove_frequency = 1
    ind_diffraction_order = 2

    unit_groove_frequency = 1 / u.um
    unit_diffraction_order = u.dimensionless_unscaled

    zemax_surface = zemax_system.LDE.GetSurfaceAt(surface_index)
    zemax_surface.ChangeType(zemax_surface.GetSurfaceTypeSettings(ZOSAPI.Editors.LDE.SurfaceType.DiffractionGrating))

    util.set_float(zemax_system, surface.groove_frequency, configuration_shape, op_groove_frequency,
                   unit_groove_frequency, surface_index, ind_groove_frequency)
    util.set_float(zemax_system, surface.diffraction_order, configuration_shape, op_diffraction_order, None,
                   surface_index, ind_diffraction_order)

    if isinstance(surface, system.surface.EllipticalGrating1):
        elliptical_grating.add_to_zemax_system(zemax_system, surface, surface_index, configuration_shape, zemax_units)
