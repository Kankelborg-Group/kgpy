import abc
import dataclasses
import typing as typ
import astropy.units as u
from kgpy.optics.system import coordinate
from ... import configuration
from .parent import ParentT, ParentBase

__all__ = ['Decenter']


@dataclasses.dataclass
class OperandBase:
    _x_op: configuration.SurfaceOperand = dataclasses.field(default=None, init=False, repr=False)
    _y_op: configuration.SurfaceOperand = dataclasses.field(default=None, init=False, repr=False)


@dataclasses.dataclass
class Base:

    parent: typ.Any = None


@dataclasses.dataclass
class Decenter(typ.Generic[ParentT], Base, coordinate.Decenter, OperandBase,):

    def _update(self) -> typ.NoReturn:
        self.x = self.x
        self.y = self.y

    @abc.abstractmethod
    def _x_setter(self, value: float):
        pass

    @abc.abstractmethod
    def _y_setter(self, value: float):
        pass

    @property
    def x(self) -> u.Quantity:
        return self._x

    @x.setter
    def x(self, value: u.Quantity):
        self._x = value
        try:
            self._x_op.surface_index = self.parent.parent.lde_index
            self.parent.parent.lde.system.set(value, self._x_setter, self._x_op, self.parent.parent.lens_units)
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
            self.parent.parent.lde.system.set(value, self._y_setter, self._y_op, self.parent.parent.lens_units)
        except AttributeError:
            pass

    @property
    def parent(self) -> ParentT:
        return self._parent

    @parent.setter
    def parent(self, value: ParentT):
        self._parent = value
        self._update()
