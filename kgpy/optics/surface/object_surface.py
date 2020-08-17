import dataclasses
import typing as typ
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from kgpy.vector import z
from .. import coordinate, Rays
from . import Surface

__all__ = ['ObjectSurface']


@dataclasses.dataclass
class ObjectSurface(Surface):

    rays_input: typ.Optional[Rays] = None

    def to_zemax(self):
        raise NotImplementedError

    @property
    def thickness_eff(self) -> u.Quantity:
        if not np.isfinite(self.thickness):
            return 0 * self.thickness
        return self.thickness

    def normal(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        return u.Quantity([0, 0, 1])

    @property
    def _rays_output(self) -> typ.Optional[Rays]:
        if self.rays_input is None:
            return None
        rays = self.rays_input.copy()
        if np.isfinite(self.thickness):
            rays.position[z] -= self.thickness
        return rays

    def sag(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        return 0 * u.mm

    @property
    def pre_transform(self) -> coordinate.Transform:
        return coordinate.Transform()

    @property
    def post_transform(self) -> coordinate.Transform:
        return coordinate.Transform()

    def copy(self) -> 'ObjectSurface':
        rays = self.rays_input
        if rays is not None:
            rays = rays.copy()
        return ObjectSurface(
            name=self.name.copy(),
            thickness=self.thickness.copy(),
            is_active=self.is_active.copy(),
            is_visible=self.is_visible.copy(),
            rays_input=rays,
        )
