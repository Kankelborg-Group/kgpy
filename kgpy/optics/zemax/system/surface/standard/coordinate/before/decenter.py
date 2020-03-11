import dataclasses
import typing as typ
import astropy.units as u

from kgpy.optics.system import coordinate
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import configuration
from .tilt_decenter import TiltDecenter

__all__ = ['Tilt']


@dataclasses.dataclass
class InstanceVarBase:
    _x_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.CBDX,
        ),
        init=None,
        repr=None,
    )
    _y_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.CBDY,
        ),
        init=None,
        repr=None,
    )


@dataclasses.dataclass
class Base(coordinate.Decenter, InstanceVarBase):
    tilt_decenter: typ.Optional[TiltDecenter] = None


class Tilt(Base):

    def _update(self):
        self.x = self.x
        self.y = self.y

    def _x_setter(self, value: float):
        self.tilt_decenter.surface.lde_row.TiltDecenterData.BeforeSurfaceDecenterX = value

    def _y_setter(self, value: float):
        self.tilt_decenter.surface.lde_row.TiltDecenterData.BeforeSurfaceDecenterY = value

    @property
    def x(self) -> u.Quantity:
        return self._x

    @x.setter
    def x(self, value: u.Quantity):
        self._x = value
        try:
            self._x_op.surface_index = self.tilt_decenter.surface.lde_index
            self.tilt_decenter.surface.lde.system.set(value, self._x_setter, self._x_op,
                                                      self.tilt_decenter.surface.lens_units)
        except AttributeError:
            pass

    @property
    def y(self) -> u.Quantity:
        return self._y

    @y.setter
    def y(self, value: u.Quantity):
        self._y = value
        try:
            self._y_op.surface_index = self.tilt_decenter.surface.lde_index
            self.tilt_decenter.surface.lde.system.set(value, self._y_setter, self._y_op,
                                                      self.tilt_decenter.surface.lens_units)
        except AttributeError:
            pass

    @property
    def tilt_decenter(self) -> TiltDecenter:
        return self._tilt_decenter

    @tilt_decenter.setter
    def tilt_decenter(self, value: TiltDecenter):
        self._tilt_decenter = value
        self._update()
