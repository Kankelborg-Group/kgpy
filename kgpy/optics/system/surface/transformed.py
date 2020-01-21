import dataclasses
import typing as typ
import astropy.units as u

import kgpy.typing.numpy as npt

from .. import mixin
from . import Surface, SurfacesRelative, Standard

__all__ = ['Mechanical']

SurfaceType = typ.TypeVar('SurfaceType')


@dataclasses.dataclass
class Mechanical(mixin.Named, typ.Generic[SurfaceType]):
    """
    A representation of an optical surface and a mechanical aperture with two main transformations and an error
    transformation.

    """

    main_surface: SurfaceType
    aperture_surface: SurfaceType

    tilt_1: u.Quantity = dataclasses.field(default_factory=lambda: [0, 0, 0] * u.deg)
    tilt_2: u.Quantity = dataclasses.field(default_factory=lambda: [0, 0, 0] * u.deg)
    tilt_err: u.Quantity = dataclasses.field(default_factory=lambda: [0, 0, 0] * u.deg)

    decenter_1: u.Quantity = dataclasses.field(default_factory=lambda: [0, 0, 0] * u.m)
    decenter_2: u.Quantity = dataclasses.field(default_factory=lambda: [0, 0, 0] * u.m)
    decenter_err: u.Quantity = dataclasses.field(default_factory=lambda: [0, 0, 0] * u.m)

    tilt_first_1: typ.Union[bool, npt.Array[bool]] = False
    tilt_first_2: typ.Union[bool, npt.Array[bool]] = False
    tilt_first_err: typ.Union[bool, npt.Array[bool]] = False

    @property
    def surfaces(self):
        return [self.aperture_surface, self.main_surface]

    @property
    def all_surfaces(self) -> typ.List[Surface]:

        name_1 = self.name + '.transform_1'
        name_2 = self.name + '.transform_2'
        name_err = self.name + '.transform_err'

        s_err = SurfacesRelative(name_err, self.tilt_err, self.decenter_err, self.tilt_first_err, self.surfaces)
        s_2 = SurfacesRelative(name_2, self.tilt_2, self.decenter_2, self.tilt_first_2, s_err.all_surfaces)
        s_1 = SurfacesRelative(name_1, self.tilt_1, self.decenter_1, self.tilt_first_1, s_2.all_surfaces)

        return s_1.all_surfaces
