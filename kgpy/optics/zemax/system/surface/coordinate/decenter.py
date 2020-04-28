import abc
import dataclasses
import typing as typ
import astropy.units as u
from kgpy.component import Component
from kgpy.optics.system.surface import coordinate
from ... import configuration, surface

__all__ = ['Decenter']

SurfaceComponentT = typ.TypeVar('SurfaceComponentT', bound='Component[surface.Surface]')


@dataclasses.dataclass
class OperandBase:
    _x_op: configuration.SurfaceOperand = dataclasses.field(default=None, init=False, repr=False)
    _y_op: configuration.SurfaceOperand = dataclasses.field(default=None, init=False, repr=False)


@dataclasses.dataclass
class Decenter(Component[SurfaceComponentT], coordinate.Decenter, OperandBase, abc.ABC):

    def _update(self) -> typ.NoReturn:
        super()._update()
        self.x = self.x
        self.y = self.y

    @abc.abstractmethod
    def _x_setter(self, value: float):
        pass

    @property
    def x(self) -> u.Quantity:
        return self._x

    @x.setter
    def x(self, value: u.Quantity):
        self._x = value
        try:
            self._composite._composite._set_with_lens_units(value, self._x_setter, self._x_op)
        except AttributeError:
            pass

    @abc.abstractmethod
    def _y_setter(self, value: float):
        pass

    @property
    def y(self) -> u.Quantity:
        return self._y

    @y.setter
    def y(self, value: u.Quantity):
        self._y = value
        try:
            self._composite._composite._set_with_lens_units(value, self._y_setter, self._y_op)
        except AttributeError:
            pass
