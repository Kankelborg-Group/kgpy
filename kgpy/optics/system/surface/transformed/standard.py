import dataclasses
import typing as typ
import numpy as np
import astropy.units as u

import kgpy.typing.numpy as npt

from kgpy.optics.system import mixin
from kgpy.optics.system.surface import Surface, SurfacesRelative, standard, Material, Aperture

__all__ = ['Standard']

ApertureType = typ.TypeVar('ApertureType')


@dataclasses.dataclass
class Standard(standard.Standard):
    """
    A representation of an optical surface and a mechanical aperture with two main transformations and an error
    transformation.

    """

    mechanical_aperture: typ.Optional[Aperture] = None

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
    def aperture_surface(self) -> standard.Standard:

        pass
    @property
    def surfaces(self) -> typ.List[Surface]:
        return [self.aperture_surface, self]

    @property
    def all_surfaces(self) -> typ.List[Surface]:

        name_1 = self.name + '.transform_1'
        name_2 = self.name + '.transform_2'
        name_err = self.name + '.transform_err'

        s_err = SurfacesRelative(name_err, self.tilt_err, self.decenter_err, self.tilt_first_err, self.surfaces)
        s_2 = SurfacesRelative(name_2, self.tilt_2, self.decenter_2, self.tilt_first_2, s_err.all_surfaces)
        s_1 = SurfacesRelative(name_1, self.tilt_1, self.decenter_1, self.tilt_first_1, s_2.all_surfaces)

        return s_1.all_surfaces
