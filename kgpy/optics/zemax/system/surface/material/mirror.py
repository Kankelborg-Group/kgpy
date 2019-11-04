import typing as tp
import pathlib
from astropy import units as u

from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import util

__all__ = ['add_to_zemax_surface']


def add_to_zemax_surface(
        zemax_system: ZOSAPI.IOpticalSystem,
        material: 'system.surface.material.Mirror',
        surface_index: int,
        zemax_units: u.Unit,
):

    zemax_surface = zemax_system.LDE.GetSurfaceAt(surface_index)
    zemax_surface.DrawData.MirrorThickness = material.thickness.to(zemax_units).value
    zemax_surface.DrawData.MirrorSubstrate = ZOSAPI.Editors.LDE.SubstrateType.Flat
