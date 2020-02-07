import dataclasses
import typing as typ

from ... import Name, mixin, coordinate
from .. import Surface, Standard, CoordinateTransform

SurfacesT = typ.TypeVar('SurfacesT')


@dataclasses.dataclass
class Single(mixin.Named, typ.Generic[SurfacesT]):
    """
    This object lets you place a list of surfaces relative to the position of the current surface, and then return to
    the position of the current surface after the list of surfaces.
    This is useful if an optical component is represented as more than one surface and each surface needs to move in
    tandem.
    """

    surfaces: SurfacesT = None
    _transform_before: CoordinateTransform = None
    _transform_after: CoordinateTransform = None
    is_last_surface: bool = False

    def __post_init__(self):

        if self._transform_before is None:
            self._transform_before = CoordinateTransform(name=Name(self.name, 'transform_before'))

        if self._transform_after is None:
            self._transform_after = CoordinateTransform(name=Name(self.name, 'transform_after'))

    @classmethod
    def from_properties(
            cls,
            name: Name,
            main: SurfacesT,
            transform: typ.Optional[coordinate.Transform] = None
    ):

        if transform is None:
            transform = coordinate.Transform()

        s = cls(name, main)
        s.transform = transform

        return s

    @property
    def transform(self) -> coordinate.Transform:
        return self._transform_before.transform

    @transform.setter
    def transform(self, value: coordinate.Transform):
        self._transform_before.transform = value
        self._transform_after.transform = ~value

    def __iter__(self):

        yield from self._transform_before

        yield from self.surfaces

        if not self.is_last_surface:
            yield from self._transform_after
