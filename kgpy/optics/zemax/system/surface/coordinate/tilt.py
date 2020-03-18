import abc
import dataclasses
import typing as typ
import astropy.units as u
from kgpy.optics.system import coordinate
from .parent import ParentT, ParentBase
from ... import configuration

__all__ = ['Tilt']


@dataclasses.dataclass
class OperandBase:
    _x_op: configuration.SurfaceOperand = dataclasses.field(default=None, init=False, repr=False)
    _y_op: configuration.SurfaceOperand = dataclasses.field(default=None, init=False, repr=False)
    _z_op: configuration.SurfaceOperand = dataclasses.field(default=None, init=False, repr=False)
    _unit: typ.ClassVar[u.Unit] = u.deg


@dataclasses.dataclass
class Tilt(typ.Generic[ParentT], ParentBase[ParentT], coordinate.Tilt, OperandBase, ):

    def _update(self) -> typ.NoReturn:
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
            self._x_op.surface_index = self.parent.parent.lde_index
            self.parent.parent.lde.system.set(value, self._x_setter, self._x_op, self._unit)
        except AttributeError:
            pass

    @property
    def y(self) -> u.Quantity:
        return self._y

    @y.setter
    def y(self, value: u.Quantity):
        self._y = value
        try:
            self._y_op.surface_index = self.parent.parent.lde_index
            self.parent.parent.lde.system.set(value, self._y_setter, self._y_op, self._unit)
        except AttributeError:
            pass

    @property
    def z(self) -> u.Quantity:
        return self._z

    @z.setter
    def z(self, value: u.Quantity):
        self._z = value
        try:
            self._z_op.surface_index = self.parent.parent.lde_index
            self.parent.parent.lde.system.set(value, self._z_setter, self._z_op, self._unit)
        except AttributeError:
            pass

    @property
    def parent(self) -> ParentT:
        return self._parent

    @parent.setter
    def parent(self, value: ParentT):
        self._parent = value
        self._update()
