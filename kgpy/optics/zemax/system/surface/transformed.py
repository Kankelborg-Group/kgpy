import dataclasses
import typing as typ
import kgpy.optics.system.surface
from . import Surface, CoordinateTransform

__all__ = ['Transformed']

SurfacesT = typ.TypeVar('SurfacesT', bound=typ.Union[typ.Iterable[Surface]])


@dataclasses.dataclass
class Transform(kgpy.optics.system.surface.transformed.coordinate.Transform):
    _ct1: CoordinateTransform = dataclasses.field(default_factory=lambda: CoordinateTransform(), init=False, repr=False)
    _ct2: CoordinateTransform = dataclasses.field(default_factory=lambda: CoordinateTransform(), init=False, repr=False)


@dataclasses.dataclass
class TransformList(kgpy.optics.system.surface.transformed.TransformList):
    _list: typ.List[Transform] = dataclasses.field(default_factory=lambda: [], init=False, repr=False)

    def append(self, value: Transform):
        if not isinstance(value, Transform):
            value = Transform.promote(value)
        super().append(value)

    def __setitem__(self, key: int, value: Transform):
        if not isinstance(value, Transform):
            value = Transform.promote(value)
        super().__setitem__(key, value)


@dataclasses.dataclass
class Base:
    transforms: TransformList = dataclasses.field(default_factory=lambda: TransformList())


@dataclasses.dataclass
class Transformed(Base, kgpy.optics.system.surface.Transformed[SurfacesT]):

    @property
    def transforms(self) -> TransformList:
        return self._transforms

    @transforms.setter
    def transforms(self, value: TransformList):
        if not isinstance(value, TransformList):
            value = TransformList.promote(value)
        self._transforms = value
