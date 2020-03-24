import dataclasses
import typing as typ
import numpy as np
from astropy import units as u

from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import util
from .. import surface
from . import decenterable, aperture

__all__ = ['add_to_zemax_surface']


@dataclasses.dataclass
class Decenter(surface.coordinate.Decenter['Rectangular'], decenterable.Operands):

    def _x_setter(self, value: float):
        if self.parent.is_obscuration:
            self.parent.parent.lde_row.ApertureData.CurrentTypeSettings._S_RectangularObscuration.ApertureXDecenter = value
        else:
            self.parent.parent.lde_row.ApertureData.CurrentTypeSettings._S_RectangularAperture.ApertureXDecenter = value

    def _y_setter(self, value: float):
        if self.parent.is_obscuration:
            self.parent.parent.lde_row.ApertureData.CurrentTypeSettings._S_RectangularObscuration.ApertureYDecenter = value
        else:
            self.parent.parent.lde_row.ApertureData.CurrentTypeSettings._S_RectangularAperture.ApertureYDecenter = value


@dataclasses.dataclass
class Base:

    decenter: surface.coordinate.Decenter = dataclasses.field(default_factory=lambda: Decenter())


@dataclasses.dataclass
class Rectangular(Base, system.surface.aperture.Rectangular, aperture.Aperture, ):

    def _update(self) -> typ.NoReturn:
        self.is_obscuration = self.is_obscuration
        self.decenter = self.decenter
        self.half_width_x = self.half_width_x
        self.half_width_y = self.half_width_y

    @property
    def is_obscuration(self) -> 'np.ndarray[bool]':
        return self._is_obscuration

    @is_obscuration.setter
    def is_obscuration(self, value: 'np.ndarray[bool]'):
        self._is_obscuration = value







def add_to_zemax_surface(
        zemax_system: ZOSAPI.IOpticalSystem,
        aperture: 'system.surface.aperture.Rectangular',
        surface_index: int,
        configuration_shape: typ.Tuple[int],
        zemax_units: u.Unit,
):

    if aperture.is_obscuration:
        type_ind = ZOSAPI.Editors.LDE.SurfaceApertureTypes.RectangularObscuration
    else:
        type_ind = ZOSAPI.Editors.LDE.SurfaceApertureTypes.RectangularAperture

    op_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.APTP
    op_half_width_x = ZOSAPI.Editors.MCE.MultiConfigOperandType.APMN
    op_half_width_y = ZOSAPI.Editors.MCE.MultiConfigOperandType.APMX

    unit_half_width_x = zemax_units
    unit_half_width_y = zemax_units

    util.set_int(zemax_system, type_ind, configuration_shape, op_type, surface_index)
    util.set_float(zemax_system, aperture.half_width_x, configuration_shape, op_half_width_x, unit_half_width_x,
                   surface_index)
    util.set_float(zemax_system, aperture.half_width_y, configuration_shape, op_half_width_y, unit_half_width_y,
                   surface_index)
