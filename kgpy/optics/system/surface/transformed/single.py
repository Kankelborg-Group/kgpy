import dataclasses
import typing as typ

from ... import Name, mixin, coordinate
from .. import Surface, Standard, CoordinateTransform

SurfacesT = typ.TypeVar('SurfacesT')


@dataclasses.dataclass
class Base(mixin.Named, typ.Generic[SurfacesT]):
    _transform_before: CoordinateTransform = dataclasses.field(init=False, repr=False,
                                                               default_factory=CoordinateTransform())
    _transform_after: CoordinateTransform = dataclasses.field(init=False, repr=False,
                                                              default_factory=CoordinateTransform())

    surfaces: SurfacesT = None
    transform: coordinate.Transform = dataclasses.field(default_factory=coordinate.Transform())
    is_last_surface: bool = False

    def __post_init__(self):
        self._transform_before.name = self.name + 'before'
        self._transform_after.name = self.name + 'after'

    def __iter__(self):

        yield from self._transform_before

        yield from self.surfaces

        if not self.is_last_surface:
            yield from self._transform_after


class Single(Base[SurfacesT]):
    """
    This object lets you place a list of surfaces relative to the position of the current surface, and then return to
    the position of the current surface after the list of surfaces.
    This is useful if an optical component is represented as more than one surface and each surface needs to move in
    tandem.
    """
    
    def __post_init__(self):
        
        super().__post_init__()
        
        self.transform = self.transform

    @property
    def transform(self) -> coordinate.Transform:
        return self._transform_before.transform

    @transform.setter
    def transform(self, value: coordinate.Transform):
        self._transform_before.transform = value
        self._transform_after.transform = coordinate.InverseTransform(value)


