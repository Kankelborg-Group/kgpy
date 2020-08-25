import abc
import dataclasses
import typing as typ
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
from kgpy import mixin, vector, transform
from kgpy.vector import x, y, z

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

    def plot_2d(
            self,
            ax: plt.Axes,
            sag: typ.Optional[typ.Callable[[u.Quantity, u.Quantity], u.Quantity]] = None,
            rigid_transform: typ.Optional[transform.rigid.Transform] = None,
            components: typ.Tuple[int, int] = (vector.ix, vector.iy),
    ):
        with astropy.visualization.quantity_support():
            c1, c2 = components
            wire = self.wire
            if sag is not None:
                wire[z] = wire[z] + sag(wire[x], wire[y])
            if rigid_transform is not None:
                wire = rigid_transform(wire, num_extra_dims=1)
            ax.fill(wire[..., c1].T, wire[..., c2].T, fill=False)

    def copy(self) -> 'Aperture':
        other = super().copy()      # type: Aperture
        other.num_samples = self.num_samples
        return other
