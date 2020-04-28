import dataclasses
import typing as typ
from .... import Name, mixin
from ... import coordinate, CoordinateBreak, CoordinateTransform
from . import Tilt, Translate

__all__ = ['Transform']


@dataclasses.dataclass
class Base:
    _name: Name = dataclasses.field(default_factory=lambda: Name(), init=False, repr=False)
    _ct_before: CoordinateTransform = dataclasses.field(
        default_factory=lambda: CoordinateTransform(Name('before')),
        init=False,
        repr=False,
    )
    _ct_after: CoordinateTransform = dataclasses.field(
        default_factory=lambda: CoordinateTransform(Name('after')),
        init=False,
        repr=False,
    )


@dataclasses.dataclass
class Transform(coordinate.Transform, Base):

    @property
    def name(self) -> Name:
        return self._name

    @name.setter
    def name(self, value: Name):
        self._name = value
        self._ct_before.name.parent = value
        self._ct_after.name.parent = value

    @property
    def tilt(self) -> Tilt:
        return self._tilt

    @tilt.setter
    def tilt(self, value: Tilt):
        if not isinstance(value, Tilt):
            value = Tilt.promote(value)
        value._composite = self
        self._tilt = value

    @property
    def translate(self) -> Translate:
        return self._translate

    @translate.setter
    def translate(self, value: Translate):
        if not isinstance(value, Translate):
            value = Translate.promote(value)
        value._composite = self
        self._translate = value

    @property
    def tilt_first(self) -> bool:
        return self._tilt_first

    @tilt_first.setter
    def tilt_first(self, value: bool):
        self._tilt_first = value
        self._ct_before.tilt_first = value
        self._ct_after.tilt_first = ~value

    def iter_before(self) -> typ.Iterator[CoordinateBreak]:
        yield from self._ct_before

    def iter_after(self) -> typ.Iterator[CoordinateBreak]:
        yield from self._ct_after


