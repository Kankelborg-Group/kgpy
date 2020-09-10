import abc
import dataclasses
import typing as typ
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
from kgpy import mixin, vector, transform
from .. import Sag

__all__ = ['Aperture']


@dataclasses.dataclass
class Aperture(
    mixin.Broadcastable,
    mixin.Copyable,
    abc.ABC
):
    num_samples: int = 1000

    @abc.abstractmethod
    def is_unvignetted(self, points: u.Quantity) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def min(self) -> u.Quantity:
        pass

    @property
    @abc.abstractmethod
    def max(self) -> u.Quantity:
        pass

    @property
    @abc.abstractmethod
    def wire(self) -> u.Quantity:
        pass

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[int, int] = (vector.ix, vector.iy),
            rigid_transform: typ.Optional[transform.rigid.TransformList] = None,
            sag: typ.Optional[Sag] = None,
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()

        with astropy.visualization.quantity_support():
            c1, c2 = components
            wire = self.wire
            if wire.unit.is_equivalent(u.mm):
                if sag is not None:
                    wire[vector.z] = wire[vector.z] + sag(wire[vector.x], wire[vector.y])
                if rigid_transform is not None:
                    wire = rigid_transform(wire, num_extra_dims=1)
                ax.fill(wire[..., c1].T, wire[..., c2].T, fill=False)
        return ax

    def copy(self) -> 'Aperture':
        other = super().copy()      # type: Aperture
        other.num_samples = self.num_samples
        return other
