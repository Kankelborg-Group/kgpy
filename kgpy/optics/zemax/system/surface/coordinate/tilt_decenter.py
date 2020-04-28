import dataclasses
import abc
import typing as typ
from kgpy.component import Component
from kgpy.optics.system.surface import coordinate
from ... import configuration, surface
from . import Tilt, Decenter

__all__ = ['TiltDecenter']

SurfaceT = typ.TypeVar('SurfaceT', bound='surface.Surface')


@dataclasses.dataclass
class OperandBase:
    _tilt_first_op: configuration.SurfaceOperand = dataclasses.field(default=None, init=False, repr=False)


@dataclasses.dataclass
class TiltDecenter(Component[SurfaceT], coordinate.TiltDecenter, OperandBase, abc.ABC, ):

    def _update(self) -> typ.NoReturn:
        super()._update()
        self.tilt = self.tilt
        self.decenter = self.decenter
        self.tilt_first = self.tilt_first

    @property
    def tilt(self) -> Tilt:
        return self._tilt

    @tilt.setter
    def tilt(self, value: Tilt):
        value._composite = self
        self._tilt = value

    @property
    def decenter(self) -> Decenter:
        return self._decenter

    @decenter.setter
    def decenter(self, value: Decenter):
        value._composite = self
        self._decenter = value

    @abc.abstractmethod
    def _tilt_first_setter(self, value: int):
        pass

    @property
    def tilt_first(self) -> bool:
        return self._tilt_first

    @tilt_first.setter
    def tilt_first(self, value: bool):
        self._tilt_first = value
        try:
            self._composite._set(int(value), self._tilt_first_setter, self._tilt_first_op)
        except AttributeError:
            pass
