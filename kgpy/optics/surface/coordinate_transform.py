import dataclasses
import typing as typ
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import kgpy.transform
from .. import Rays
from . import Surface

__all__ = ['CoordinateTransform']


@dataclasses.dataclass
class CoordinateTransform(Surface):

    transform: typ.Optional[kgpy.transform.rigid.Transform] = None

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.transform.config_broadcast,
        )

    def sag(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        return 0 * u.mm

    def normal(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        return u.Quantity([0, 0, 1])

    @property
    def _rays_output(self) -> typ.Optional[Rays]:
        return self.rays_input.copy()

    @property
    def pre_transform(self) -> kgpy.transform.rigid.TransformList:
        # return kgpy.transform.rigid.TransformList()
        return kgpy.transform.rigid.TransformList([self.transform])

    @property
    def post_transform(self) -> kgpy.transform.rigid.TransformList:
        # return kgpy.transform.rigid.TransformList([self.transform, kgpy.transform.rigid.Translate(z=self.thickness)])
        return kgpy.transform.rigid.TransformList([kgpy.transform.rigid.Translate.from_components(z=self.thickness)])

    def copy(self) -> 'CoordinateTransform':
        return CoordinateTransform(
            name=self.name.copy(),
            thickness=self.thickness.copy(),
            is_active=self.is_active.copy(),
            is_visible=self.is_visible.copy(),
            transform=self.transform.copy(),
        )

    def plot_2d(
            self,
            ax: plt.Axes,
            rigid_transform: typ.Optional[kgpy.transform.rigid.Transform] = None,
            components: typ.Tuple[int, int] = (0, 1),
    ) -> plt.Axes:
        pass
