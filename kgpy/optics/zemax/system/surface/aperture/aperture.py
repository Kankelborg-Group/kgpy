import dataclasses
import typing as typ
from kgpy.component import Component
from .. import standard

__all__ = ['Aperture', 'NoAperture']


@dataclasses.dataclass
class Aperture(Component[standard.Standard]):
    pass


class NoAperture(Aperture):
    def _update(self) -> typ.NoReturn:
        super()._update()
