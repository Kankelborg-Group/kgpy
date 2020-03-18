import dataclasses
import typing as typ
from .. import surface

__all__ = ['ParentT', 'ParentBase']

SurfaceT = typ.TypeVar('SurfaceT', bound='surface.Surface')


class SurfaceGrandchild(typ.Generic[SurfaceT]):

    @property
    def parent(self) -> SurfaceT:
        ...

    @parent.setter
    def parent(self, value: SurfaceT):
        ...


ParentT = typ.TypeVar('ParentT', bound='SurfaceGrandchild')


@dataclasses.dataclass
class ParentBase(typ.Generic[ParentT]):

    parent: typ.Optional[ParentT] = None
