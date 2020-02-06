import dataclasses

from kgpy.optics.system import mixin, Name, coordinate
from kgpy.optics.system.surface import CoordinateBreak


@dataclasses.dataclass
class CoordinateTransform(mixin.Named):
    """
    Zemax doesn't allow decenters in the z-direction, instead they intend this concept to be represented by the
    `thickness` parameter.
    The problem with their representation is that the `thickness` parameter is always applied last and does not respect
    the `order` parameter.
    If you're trying to invert a 3D translation/rotation this is a problem since sometimes you need the translation
    in z applied first.
    The obvious way of fixing the problem is to define another surface before the coordinate break surface that can
    apply the z-translation first if needed.
    This is a class that acts as a normal `CoordinateBreak`, but is a composition of two coordinate breaks.
    It respects the `order` parameter for 3D translations.
    """

    _cb1: CoordinateBreak = None
    _cb2: CoordinateBreak = None

    def __post_init__(self):

        if self._cb1 is None:
            self._cb1 = CoordinateBreak(name=self.name + 'cb1')

        if self._cb2 is None:
            self._cb2 = CoordinateBreak(name=self.name + 'cb2')

    @classmethod
    def from_cbreak_args(
            cls,
            name: Name,
            transform: coordinate.Transform,
    ):

        s = cls(name)
        s.transform = transform

    @property
    def transform(self) -> coordinate.Transform:
        return self._cb1.transform + self._cb2.transform

    @transform.setter
    def transform(self, value: coordinate.Transform):
        
        if value.tilt_first:

            self._cb1.transform = coordinate.Transform(tilt=value.tilt, tilt_first=value.tilt_first)
            self._cb2.transform = coordinate.Transform(decenter=value.decenter, tilt_first=value.tilt_first)

        else:
            
            self._cb1.transform = coordinate.Transform(decenter=value.decenter, tilt_first=value.tilt_first)
            self._cb2.transform = coordinate.Transform(tilt=value.tilt, tilt_first=value.tilt_first)

    def __iter__(self):
        yield self._cb1
        yield self._cb2