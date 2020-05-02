import dataclasses
import typing as typ
from kgpy.component import Component
from .... import Name, mixin, surface
from ... import coordinate, CoordinateBreak, coordinate_transform
from . import Tilt, Translate

__all__ = ['Transform']


@dataclasses.dataclass
class Base:
    _name: Name = dataclasses.field(default_factory=lambda: Name(), init=False, repr=None)
    _cb1: CoordinateBreak = dataclasses.field(default_factory=lambda: CoordinateBreak(), init=False, repr=False, )
    _cb2: CoordinateBreak = dataclasses.field(default_factory=lambda: CoordinateBreak(), init=False, repr=False, )


@dataclasses.dataclass
class Transform(Component['coordinate_transform.CoordinateTransform'], coordinate.Transform, Base):

    def _update(self) -> typ.NoReturn:
        super()._update()
        self.tilt_first = self.tilt_first

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
        try:
            self._cb1.tilt_first = value
            self._cb2.tilt_first = value
            self._cb_tilt.name.base = 'tilt'
            self._cb_translate.name.base = 'translate'
        except AttributeError:
            pass
        self.tilt = self.tilt
        self.translate = self.translate

    def __iter__(self) -> typ.Iterator[CoordinateBreak]:
        yield self._cb1
        yield self._cb2
