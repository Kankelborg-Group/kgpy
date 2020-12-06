import dataclasses
import typing as typ
import win32com.client
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
        self._composite._lde_row_aperture_data.ApertureXDecenter = value

    def _y_setter(self, value: float):
        self._composite._lde_row_aperture_data.ApertureYDecenter = value


@dataclasses.dataclass
class Operands:

    _half_width_x_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.APMN
        ),
        init=False,
        repr=None,
    )

    _half_width_y_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
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

    @property
    def _lde_row_aperture_data(self) -> ZOSAPI.Editors.LDE.ISurfaceApertureRectangular:
        if self._is_obscuration:
            aper_type = ZOSAPI.Editors.LDE.SurfaceApertureTypes.RectangularObscuration
        else:
            aper_type = ZOSAPI.Editors.LDE.SurfaceApertureTypes.RectangularAperture

        return win32com.client.CastTo(
            self._composite._lde_row.ApertureData.CreateApertureTypeSettings(aper_type),
            ZOSAPI.Editors.LDE.ISurfaceApertureRectangular.__name__
        )

    @_lde_row_aperture_data.setter
    def _lde_row_aperture_data(self, value: ZOSAPI.Editors.LDE.ISurfaceApertureRectangular):
        self._composite._lde_row.ApertureData.ChangeApertureTypeSettings(value)

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
            self._lde_row_aperture_data = self._lde_row_aperture_data
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
        s = self._lde_row_aperture_data
        s.XHalfWidth = value
        self._lde_row_aperture_data = s

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
        s = self._lde_row_aperture_data
        s.YHalfWidth = value
        self._lde_row_aperture_data = s

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
