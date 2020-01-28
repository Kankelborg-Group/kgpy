import dataclasses
import typing as typ
import numpy as np
import numpy.core

from .. import mixin, coordinate
from . import surface


__all__ = ['CoordinateBreak']


@dataclasses.dataclass
class CoordinateBreak(surface.Surface):
    """
    Representation of a Zemax Coordinate Break.
    """
    transform: coordinate.Transform = dataclasses.field(default_factory=lambda: coordinate.Transform())

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.transform.config_broadcast
        )


@dataclasses.dataclass
class ArbitraryDecenterZ(mixin.Named):
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

    _cb_main: CoordinateBreak = None
    _cb_z: CoordinateBreak = None

    def __post_init__(self):

        if self._cb_main is None:
            self._cb_main = CoordinateBreak(name=self.name)

        if self._cb_z is None:
            n = np.core.defchararray.add(self.name, np.array('_z'))
            self._cb_z = CoordinateBreak(name=n)

    @classmethod
    def from_cbreak_args(
            cls,
            name: np.ndarray[str],
            transform: coordinate.Transform,
    ):

        s = cls(name)
        s.transform = transform

    @property
    def transform(self) -> coordinate.Transform:
        return Transform.from_super(self._cb_main.transform + self._cb_z.transform, self)

    @transform.setter
    def transform(self, value: coordinate.Transform):

        if value.tilt_first:
            t_main = dataclasses.replace(value)
            t_z = dataclasses.replace(value, tilt=0*value.tilt, decenter=0*value.decenter)

        else:
            a = value.decenter
            b = a.copy()
            a[..., ~0] = 0
            b[..., :~0] = 0
            t_main = dataclasses.replace(value, decenter=a)
            t_z = dataclasses.replace(value, tilt=0*value.tilt, decenter=b)

        self._cb_main.transform = t_main
        self._cb_z = t_z

    def __iter__(self):
        yield self._cb_z
        yield self._cb_main


@dataclasses.dataclass
class Transform(coordinate.Transform, mixin.Adopted[ArbitraryDecenterZ]):

    @classmethod
    def from_super(
            cls,
            transform: coordinate.Transform,
            parent: ArbitraryDecenterZ
    ) -> 'Transform':
        return cls(parent, transform.tilt, transform.decenter, transform.tilt_first)

    @property
    def tilt_first(self) -> np.ndarray[bool]:
        return self._tilt_first

    @tilt_first.setter
    def tilt_first(self, value: np.ndarray[bool]):
        self._tilt_first = value
        self._parent.transform = self

        
    

            

