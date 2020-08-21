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

    rays_input: typ.Optional[Rays] = dataclasses.field(default=None, repr=False)

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
        rays = self.rays_input
        if rays is not None:
            rays = rays.copy()
        return rays

    def sag(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        return 0 * u.mm

    @property
    def pre_transform(self) -> coordinate.TransformList:
        return coordinate.TransformList()

    @property
    def post_transform(self) -> coordinate.TransformList:
        return coordinate.TransformList([coordinate.Translate(z=self.thickness_eff)])

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

    def plot_2d(
            self,
            ax: plt.Axes,
            transform: typ.Optional[coordinate.Transform] = None,
            components: typ.Tuple[int, int] = (0, 1),
    ) -> plt.Axes:
        pass
