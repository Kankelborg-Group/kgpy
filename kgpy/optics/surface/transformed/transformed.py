import dataclasses
import typing as typ

import kgpy.optics.system.zemax_compatible
from kgpy import mixin
from .. import Surface
from . import coordinate

__all__ = ['Transformed', 'TransformList']

SurfacesT = typ.TypeVar('SurfacesT', bound=typ.Union[typ.Iterable[Surface], kgpy.optics.system.zemax_compatible.ZemaxCompatible])


@dataclasses.dataclass
class TransformList:

    _list: typ.List[coordinate.Transform] = dataclasses.field(default_factory=lambda: [], init=False)

    @classmethod
    def promote(cls, transform_list: typ.List[coordinate.Transform]) -> 'TransformList':
        self = cls()
        for value in transform_list:
            self.append(value)
        return self

    def append(self, value: coordinate.Transform):
        if not isinstance(value, coordinate.Transform):
            value = coordinate.Transform.promote(value)
        self._list.append(value)

    def __getitem__(self, item: int):
        return self._list.__getitem__(item)

    def __setitem__(self, key: int, value: coordinate.Transform):
        if not isinstance(value, coordinate.Transform):
            value = coordinate.Transform.promote(value)
        self._list.__setitem__(key, value)

    def __len__(self):
        return self._list.__len__()


@dataclasses.dataclass
class Base:
    surfaces: SurfacesT = dataclasses.field(default_factory=lambda: [])
    transforms: TransformList = dataclasses.field(default_factory=lambda: TransformList())
    is_last_surface: bool = False


@dataclasses.dataclass
class Transformed(Base, kgpy.optics.system.zemax_compatible.ZemaxCompatible, mixin.Named, typ.Generic[SurfacesT]):

    def to_zemax(self) -> 'Transformed':
        from kgpy.optics import zemax
        return zemax.system.surface.Transformed(
            name=self.name,
            surfaces=self.surfaces.to_zemax(),
            transforms=self.transforms,
        )

    @property
    def transforms(self) -> TransformList:
        return self._transforms

    @transforms.setter
    def transforms(self, value: TransformList):
        if not isinstance(value, TransformList):
            value = TransformList.promote(value)
        self._transforms = value

    def __iter__(self) -> typ.Iterator[Surface]:

        for transform in self.transforms:
            yield from transform.iter_before()

        yield from self.surfaces

        if not self.is_last_surface:
            for transform in reversed(self.transforms):
                yield from transform.iter_after()
