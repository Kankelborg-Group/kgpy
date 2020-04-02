import abc
import dataclasses
import typing as typ
import astropy.units as u
from kgpy.component import Component
from kgpy.optics.system.surface import coordinate
from ... import configuration, surface

__all__ = ['TiltFirst']

SurfaceComponentT = typ.TypeVar('SurfaceComponentT', bound=Component[surface.Surface])


@dataclasses.dataclass
class OperandBase:
    _value_op: configuration.SurfaceOperand = dataclasses.field(default=None, init=False, repr=False)


@dataclasses.dataclass
class TiltFirst(Component[SurfaceComponentT], coordinate.TiltFirst, OperandBase, typ.Generic[SurfaceComponentT], ):

    def _update(self) -> typ.NoReturn:
        super()._update()
        self.value = self.value

    @abc.abstractmethod
    def _value_setter(self, value: float):
        pass

    @property
    def value(self) -> u.Quantity:
        return self._value

    @value.setter
    def value(self, val: u.Quantity):
        self._value = val
        try:
            self.composite.composite.set(val, self._value_setter, self._value_op)
        except AttributeError:
            pass
