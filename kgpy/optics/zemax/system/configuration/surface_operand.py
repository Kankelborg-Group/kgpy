import dataclasses
import typing as typ
from . import Operand
from .. import Child, surface as surface_

__all__ = ['SurfaceOperand']


@dataclasses.dataclass
class Base:

    surface: typ.Optional[surface_.Surface] = None


@dataclasses.dataclass
class SurfaceOperand(Base, Operand):

    def _update(self) -> typ.NoReturn:
        super()._update()
        self.surface_index = self.surface_index

    @property
    def surface_index(self) -> int:
        return self.surface.lde_index

    @surface_index.setter
    def surface_index(self, value: int):
        self.param_1 = value

    @property
    def surface(self) -> surface_.Surface:
        return self._surface

    @surface.setter
    def surface(self, value: surface_.Surface):
        self._surface = value
        self._update()
