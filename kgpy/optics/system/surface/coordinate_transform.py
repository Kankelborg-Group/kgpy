import dataclasses

from .. import mixin, Name
from . import coordinate, CoordinateBreak


@dataclasses.dataclass
class Base:

    _cb1: CoordinateBreak = dataclasses.field(
        default_factory=lambda: CoordinateBreak(name=Name('cb1')),
        init=False, 
        repr=False,
    )
    _cb2: CoordinateBreak = dataclasses.field(
        default_factory=lambda: CoordinateBreak(name=Name('cb2')),
        init=False, 
        repr=False,
    )


class CoordinateTransform(coordinate.Transform, Base, mixin.Named):
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

    @property
    def transform(self) -> coordinate.Transform:
        return coordinate.Transform(tilt=self.tilt, translate=self.translate, tilt_first=self.tilt_first)

    @transform.setter
    def transform(self, value: coordinate.Transform):

        self.tilt = value.tilt
        self.translate = value.translate
        self.tilt_first = value.tilt_first
    
    @property
    def name(self) -> Name:
        return self._cb1.name.parent
    
    @name.setter
    def name(self, value: Name):
        self._cb1.name.parent = value
        self._cb2.name.parent = value

    @property
    def tilt(self) -> coordinate.Tilt:
        if self.tilt_first.value:
            return self._cb1.tilt
        else:
            return self._cb2.tilt

    @tilt.setter
    def tilt(self, value: coordinate.Tilt):
        if self.tilt_first.value:
            self._cb1.tilt = value
            self._cb2.tilt = coordinate.Tilt()
        else:
            self._cb1.tilt = coordinate.Tilt()
            self._cb2.tilt = value

    @property
    def translate(self) -> coordinate.Translate:
        if self.tilt_first.value:
            return self._cb2.transform.translate
        else:
            return self._cb1.transform.translate

    @translate.setter
    def translate(self, value: coordinate.Translate):
        if self.tilt_first.value:
            self._cb1.transform.translate = coordinate.Translate()
            self._cb2.transform.translate = value
        else:
            self._cb1.transform.translate = value
            self._cb2.transform.translate = coordinate.Translate()

    @property
    def tilt_first(self) -> coordinate.TiltFirst:
        return self._cb1.tilt_first

    @tilt_first.setter
    def tilt_first(self, value: coordinate.TiltFirst):
        tilt = self.tilt
        translate = self.translate
        self._cb1.tilt_first = value
        self._cb2.tilt_first = value
        self.tilt = tilt
        self.translate = translate

    def __iter__(self):
        yield self._cb1
        yield self._cb2


