import dataclasses
import typing as typ

from kgpy.typing import numpy as npt

from . import coordinate_break, Standard, aperture, material
from .. import coordinate, mixin

__all__ = ['GenericSurfaces']


MainT = typ.TypeVar('MainT')


@dataclasses.dataclass
class GenericSurfaces(mixin.Named, typ.Generic[MainT]):
    """
    This object lets you place a list of surfaces relative to the position of the current surface, and then return to
    the position of the current surface after the list of surfaces.
    This is useful if an optical component is represented as more than one surface and each surface needs to move in
    tandem.
    """

    main: MainT
    cbreak_before: coordinate_break.ArbitraryDecenterZ = None
    cbreak_after: coordinate_break.ArbitraryDecenterZ = None

    def __post_init__(self):

        if self.cbreak_before is None:
            self.cbreak_before = coordinate_break.ArbitraryDecenterZ(name=self.name + '.cb_before')

        if self.cbreak_after is None:
            self.cbreak_after = coordinate_break.ArbitraryDecenterZ(name=self.name + '.cb_after')

    @classmethod
    def from_cbreak_args(
            cls,
            name: str,
            main: MainT,
            transform: typ.Optional[coordinate.Transform] = None
    ):

        if transform is None:
            transform = coordinate.Transform()

        s = cls(name, main)
        s.transform = transform

        return s

    @property
    def transform(self) -> coordinate.Transform:
        return self.cbreak_before.transform

    @transform.setter
    def transform(self, value: coordinate.Transform):
        self.cbreak_before.transform = value
        self.cbreak_after.transform = ~value

    def __iter__(self):
        for s in self.cbreak_before:
            yield s

        for s in self.main:
            yield s

        for s in self.cbreak_after:
            yield s
