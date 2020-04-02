import abc
import dataclasses
import typing as typ
import astropy.units as u
from kgpy.component import Component
from kgpy.optics.system.surface import coordinate
from ... import configuration, surface

__all__ = ['Tilt']

SurfaceChildT = typ.TypeVar('SurfaceChildT', bound=Component[surface.Surface])


@dataclasses.dataclass
class OperandBase:
    _x_op: configuration.SurfaceOperand = dataclasses.field(default=None, init=False, repr=False)
    _y_op: configuration.SurfaceOperand = dataclasses.field(default=None, init=False, repr=False)
    _z_op: configuration.SurfaceOperand = dataclasses.field(default=None, init=False, repr=False)
    _unit: typ.ClassVar[u.Unit] = u.deg


@dataclasses.dataclass
class Tilt(Component[SurfaceChildT], coordinate.Tilt, OperandBase, typ.Generic[SurfaceChildT], ):

    def _update(self) -> typ.NoReturn:
        super()._update()
        self.x = self.x
        self.y = self.y
        self.z = self.z

    @abc.abstractmethod
    def _x_setter(self, value: float):
        pass

    @abc.abstractmethod
    def _y_setter(self, value: float):
        pass

    @abc.abstractmethod
    def _z_setter(self, value: float):
        pass

    @property
    def x(self) -> u.Quantity:
        return self._x

    @x.setter
    def x(self, value: u.Quantity):
        self._x = value
        try:
            self.composite.composite.set(value, self._x_setter, self._x_op, self._unit)
        except AttributeError:
            pass

    @property
    def y(self) -> u.Quantity:
        return self._y

    @y.setter
    def y(self, value: u.Quantity):
        self._y = value
        try:
            self.composite.composite.set(value, self._y_setter, self._y_op, self._unit)
        except AttributeError:
            pass

    @property
    def z(self) -> u.Quantity:
        return self._z

    @z.setter
    def z(self, value: u.Quantity):
        self._z = value
        try:
            self.composite.composite.set(value, self._z_setter, self._z_op, self._unit)
        except AttributeError:
            pass
