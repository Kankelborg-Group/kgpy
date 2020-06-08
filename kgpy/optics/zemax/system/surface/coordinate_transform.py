import dataclasses
import kgpy.optics.system.surface
from . import CoordinateBreak

__all__ = ['CoordinateTransform']


@dataclasses.dataclass
class Transform(kgpy.optics.system.surface.coordinate_transform.coordinate.Transform):
    _cb1: CoordinateBreak = dataclasses.field(default_factory=lambda: CoordinateBreak(), init=False, repr=False, )
    _cb2: CoordinateBreak = dataclasses.field(default_factory=lambda: CoordinateBreak(), init=False, repr=False, )


@dataclasses.dataclass
class CoordinateTransform(kgpy.optics.system.surface.CoordinateTransform):

    @property
    def transform(self) -> Transform:
        return self._transform

    @transform.setter
    def transform(self, value: Transform):
        if not isinstance(value, Transform):
            value = Transform.promote(value)
        self._transform = value
