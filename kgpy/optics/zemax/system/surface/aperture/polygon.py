import typing as tp
import pathlib
import hashlib
from astropy import units as u

from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import util
from . import uda

__all__ = ['add_to_zemax_surface']


def add_to_zemax_surface(
        zemax_system: ZOSAPI.IOpticalSystem,
        aperture: 'system.surface.aperture.Polygon',
        surface_index: int,
        configuration_shape: tp.Tuple[int],
        zemax_units: u.Unit,
):
    if aperture.is_obscuration:
        type_ind = ZOSAPI.Editors.LDE.SurfaceApertureTypes.UserObscuration
    else:
        type_ind = ZOSAPI.Editors.LDE.SurfaceApertureTypes.UserAperture

    op_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.APTP
    op_file = ZOSAPI.Editors.MCE.MultiConfigOperandType.UDAF

    filename = hashlib.md5(aperture.points.tobytes()).hexdigest()[:10] + '.uda'
    aper_path = pathlib.Path('Apertures')
    filepath = pathlib.Path(zemax_system.TheApplication.ObjectsDir) / aper_path / filename

    uda.write_file(filepath, aperture.points, zemax_units)

    util.set_int(zemax_system, type_ind, configuration_shape, op_type, surface_index)
    util.set_str(zemax_system, str(filename), configuration_shape, op_file, surface_index)


