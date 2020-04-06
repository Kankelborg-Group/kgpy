import dataclasses

from ... import mixin, Name, surface
from .. import CoordinateBreak
from . import coordinate

__all__ = ['CoordinateTransform']


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

    transform: coordinate.Transform = dataclasses.field(default_factory=lambda: coordinate.Transform(), init=False,
                                                        repr=False)


class CoordinateTransform(surface.coordinate.Transform, Base, mixin.Named):
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
    def _cb_tilt(self) -> CoordinateBreak:
        if self.tilt_first:
            return self._cb1
        else:
            return self._cb2

    @property
    def _cb_translate(self) -> CoordinateBreak:
        if self.tilt_first:
            return self._cb2
        else:
            return self._cb1

    @property
    def name(self) -> Name:
        return self._name

    @name.setter
    def name(self, value: Name):
        self._name = value
        self._cb1.name.parent = value
        self._cb2.name.parent = value

    @property
    def transform(self) -> coordinate.Transform:
        return self._transform

    @transform.setter
    def transform(self, value: coordinate.Transform):
        value._composite = self
        self._transform = value

    @property
    def tilt(self) -> coordinate.Tilt:
        return self.transform.tilt

    @tilt.setter
    def tilt(self, value: coordinate.Tilt):
        self.transform.tilt = value

    @property
    def translate(self) -> coordinate.Translate:
        return self.transform.translate

    @translate.setter
    def translate(self, value: coordinate.Translate):
        self.transform.translate = value

    @property
    def tilt_first(self) -> bool:
        return self.transform.tilt_first

    @tilt_first.setter
    def tilt_first(self, value: bool):
        self.transform.tilt_first = value

    def __iter__(self):
        yield self._cb1
        yield self._cb2
