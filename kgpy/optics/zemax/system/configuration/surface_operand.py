import dataclasses
import typing as typ
from . import Operand
from .. import surface as surface_

__all__ = ['SurfaceOperand']


@dataclasses.dataclass
class Base:

    surface: 'typ.Optional[surface_.Surface]' = None


@dataclasses.dataclass
class SurfaceOperand(Base, Operand):

    def _update(self) -> typ.NoReturn:
        super()._update()
        try:
            self.param_1 = self.surface._lde_index
        except AttributeError:
            pass

    @property
    def surface(self) -> 'surface_.Surface':
        return self._surface

    @surface.setter
    def surface(self, value: 'surface_.Surface'):
        self._surface = value
        self._update()
