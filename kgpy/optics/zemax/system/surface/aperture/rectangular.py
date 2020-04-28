import dataclasses
import typing as typ
import numpy as np
from astropy import units as u

from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import util, configuration
from ... import surface
from . import decenterable, aperture

__all__ = ['Rectangular']


@dataclasses.dataclass
class Decenter(surface.coordinate.Decenter['Rectangular'], decenterable.Operands):

    def _x_setter(self, value: float):
        self._composite._composite._lde_row.ApertureData.ApertureXDecenter = value

    def _y_setter(self, value: float):
        self._composite._composite._lde_row.ApertureData.ApertureYDecenter = value


@dataclasses.dataclass
class Operands:

    _half_width_x_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.APMN
        ),
        init=False,
        repr=None,
    )

    _half_width_y_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.APMN
        ),
        init=False,
        repr=None,
    )


@dataclasses.dataclass
class Base:

    decenter: Decenter = dataclasses.field(default_factory=lambda: Decenter())


@dataclasses.dataclass
class Rectangular(Base, system.surface.aperture.Rectangular, aperture.Aperture, Operands):

    def _update(self) -> typ.NoReturn:
        super()._update()
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
        try:
            if self._is_obscuration:
                aper_type = ZOSAPI.Editors.LDE.SurfaceApertureTypes.RectangularObscuration
            else:
                aper_type = ZOSAPI.Editors.LDE.SurfaceApertureTypes.RectangularAperture
            aper_type = self._composite._lde_row.ApertureData.CreateApertureTypeSettings(aper_type)
            self._composite._lde_row.ApertureData.ChangeApertureTypeSettings(aper_type)
        except AttributeError:
            pass

    @property
    def decenter(self) -> Decenter:
        return self._decenter

    @decenter.setter
    def decenter(self, value: Decenter):
        if not isinstance(value, Decenter):
            value = Decenter(value.x, value.y)
        self._decenter = value

    def _half_width_x_setter(self, value: float):
        self._composite._lde_row.ApertureData.HalfWidthX = value

    @property
    def half_width_x(self) -> u.Quantity:
        return self._half_width_x

    @half_width_x.setter
    def half_width_x(self, value: u.Quantity):
        self._half_width_x = value
        try:
            self._composite._set_with_lens_units(value, self._half_width_x_setter, self._half_width_x_op)
        except AttributeError:
            pass

    def _half_width_y_setter(self, value: float):
        self._composite._lde_row.ApertureData.HalfWidthY = value

    @property
    def half_width_y(self) -> u.Quantity:
        return self._half_width_y

    @half_width_y.setter
    def half_width_y(self, value: u.Quantity):
        self._half_width_y = value
        try:
            self._composite._set_with_lens_units(value, self._half_width_y_setter, self._half_width_y_op)
        except AttributeError:
            pass
