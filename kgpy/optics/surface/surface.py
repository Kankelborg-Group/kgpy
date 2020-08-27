import abc
import dataclasses
import typing as typ
import warnings
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from kgpy import transform, mixin, optics

__all__ = ['Surface']


@dataclasses.dataclass
class Surface(
    mixin.Broadcastable,
    mixin.Named,
    mixin.Copyable,
    abc.ABC
):
    """
    This class represents a single optical surface. This class should be a drop-in replacement for a Zemax surface, and
    have all the same properties and behaviors.
    """
    thickness: u.Quantity = 0 * u.mm
    is_active: bool = True

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.thickness,
            self.is_active,
        )

    @abc.abstractmethod
    def sag(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        pass

    @abc.abstractmethod
    def normal(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        pass

    @abc.abstractmethod
    def propagate_rays(self, rays: optics.Rays) -> optics.Rays:
        pass

    @property
    @abc.abstractmethod
    def pre_transform(self) -> transform.rigid.TransformList:
        pass

    @property
    @abc.abstractmethod
    def post_transform(self) -> transform.rigid.TransformList:
        pass

    @property
    def transform_total(self) -> transform.rigid.TransformList:
        return self.pre_transform + self.post_transform

    @abc.abstractmethod
    def plot_2d(
            self,
            ax: plt.Axes,
            rigid_transform: typ.Optional[transform.rigid.Transform] = None,
            components: typ.Tuple[int, int] = (0, 1),
    ) -> plt.Axes:
        pass

    def copy(self) -> 'Copyable':
        other = super().copy()      # type: Surface
        other.thickness = self.thickness.copy()
        other.is_active = self.is_active
        return other
