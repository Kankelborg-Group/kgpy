import dataclasses
import typing as typ
from ... import mixin, Name, surface
from .. import CoordinateBreak
from . import coordinate

__all__ = ['CoordinateTransform']


@dataclasses.dataclass
class Base:
    transform: coordinate.Transform = dataclasses.field(default_factory=lambda: coordinate.Transform())


@dataclasses.dataclass
class CoordinateTransform(mixin.ZemaxCompatible, Base, mixin.Named):
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

    def __post_init__(self):
        self.name = self.name

    def to_zemax(self) -> 'CoordinateTransform':
        from kgpy.optics import zemax
        return zemax.system.surface.CoordinateTransform(
            name=self.name,
            transform=self.transform,
        )


    @property
    def name(self) -> Name:
        return self._name

    @name.setter
    def name(self, value: Name):
        self._name = value
        try:
            self.transform.name = value
        except AttributeError:
            pass

    @property
    def transform(self) -> coordinate.Transform:
        return self._transform

    @transform.setter
    def transform(self, value: coordinate.Transform):
        if not isinstance(value, coordinate.Transform):
            value = coordinate.Transform.promote(value)
        self._transform = value
        value._composite = self

    def __iter__(self) -> typ.Iterator[CoordinateBreak]:
        return self.transform.__iter__()
