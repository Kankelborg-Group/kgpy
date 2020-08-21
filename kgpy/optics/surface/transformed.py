import dataclasses
import typing as typ
import kgpy.mixin
from .. import coordinate
from . import Surface, CoordinateTransform

__all__ = ['Transformed']

SurfacesT = typ.TypeVar('SurfacesT', bound=typ.Iterable[Surface])


@dataclasses.dataclass
class Transformed(
    kgpy.mixin.Named,
    typ.Generic[SurfacesT],
):
    surfaces: SurfacesT = dataclasses.field(default_factory=lambda: [])
    transform: coordinate.Transform = dataclasses.field(default_factory=lambda: coordinate.TransformList())
    is_last_surface: bool = False

    def __iter__(self) -> typ.Iterator[Surface]:
        yield CoordinateTransform(name=self.name + 'transform', transform=self.transform)
        yield from self.surfaces
        if not self.is_last_surface:
            yield CoordinateTransform(name=self.name + 'untransform', transform=~self.transform)
