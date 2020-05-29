import dataclasses
import typing as typ
import kgpy
from .... import coordinate as base_coordinate
from ... import CoordinateBreak, CoordinateTransform
from . import Tilt, Translate

__all__ = ['Transform']


@dataclasses.dataclass
class Base:
    _name: kgpy.Name = dataclasses.field(default_factory=lambda: kgpy.Name(), init=False, repr=False)
    _ct_before: CoordinateTransform = dataclasses.field(
        default_factory=lambda: CoordinateTransform(kgpy.Name('before')),
        init=False,
        repr=False,
    )
    _ct_after: CoordinateTransform = dataclasses.field(
        default_factory=lambda: CoordinateTransform(kgpy.Name('after')),
        init=False,
        repr=False,
    )


@dataclasses.dataclass
class Transform(base_coordinate.Transform, Base):

    @property
    def name(self) -> kgpy.Name:
        return self._name

    @name.setter
    def name(self, value: kgpy.Name):
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
        self._ct_before.transform.tilt_first = value
        self._ct_after.transform.tilt_first = ~value

    def iter_before(self) -> typ.Iterator[CoordinateBreak]:
        yield from self._ct_before

    def iter_after(self) -> typ.Iterator[CoordinateBreak]:
        yield from self._ct_after


