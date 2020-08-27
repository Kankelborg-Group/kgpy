import dataclasses
import typing as typ
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from kgpy import transform, optics
from . import Surface

__all__ = ['ObjectSurface']


@dataclasses.dataclass
class ObjectSurface(Surface):

    @property
    def thickness_eff(self) -> u.Quantity:
        if np.isfinite(self.thickness):
            return self.thickness
        return 0 * u.mm

    def normal(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        return u.Quantity([0, 0, 1])

    def propagate_rays(self, rays: optics.Rays) -> optics.Rays:
        return rays.copy()

    def sag(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        return 0 * u.mm

    @property
    def pre_transform(self) -> transform.rigid.TransformList:
        return transform.rigid.TransformList()

    @property
    def post_transform(self) -> transform.rigid.TransformList:
        return transform.rigid.TransformList([transform.rigid.Translate.from_components(z=self.thickness_eff)])

    def copy(self) -> 'ObjectSurface':
        other = super().copy()      # type: ObjectSurface
        if self.rays_input is None:
            other.rays = self.rays_input
        else:
            other.rays = self.rays_input.copy()
        return other

    def plot_2d(
            self,
            ax: plt.Axes,
            rigid_transform: typ.Optional[transform.rigid.Transform] = None,
            components: typ.Tuple[int, int] = (0, 1),
    ) -> plt.Axes:
        pass
