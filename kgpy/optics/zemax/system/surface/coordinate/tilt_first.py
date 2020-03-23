import abc
import dataclasses
import typing as typ
import astropy.units as u
from kgpy.optics.system.surface import coordinate
from ... import Child, configuration
from ..surface import SurfaceChildT

__all__ = ['TiltFirst']


@dataclasses.dataclass
class OperandBase:
    _value_op: configuration.SurfaceOperand = dataclasses.field(default=None, init=False, repr=False)


@dataclasses.dataclass
class TiltFirst(Child[SurfaceChildT], coordinate.TiltFirst, OperandBase, typ.Generic[SurfaceChildT], ):

    def _update(self) -> typ.NoReturn:
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
            self._value_op.surface_index = self.parent.parent.lde_index
            self.parent.parent.lde.system.set(val, self._value_setter, self._value_op)
        except AttributeError:
            pass
