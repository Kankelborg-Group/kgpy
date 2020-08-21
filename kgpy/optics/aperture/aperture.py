import abc
import dataclasses
import typing as typ
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import kgpy.mixin
import kgpy.vector
from kgpy.vector import x, y, z
from .. import coordinate

__all__ = ['Aperture']


@dataclasses.dataclass
class Aperture(
    kgpy.mixin.Copyable,
    kgpy.mixin.Broadcastable,
    abc.ABC
):
    num_samples: int = 1000
    is_active: bool = True
    is_test_stop: bool = True

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
            transform: typ.Optional[coordinate.Transform] = None,
            components: typ.Tuple[int, int] = (kgpy.vector.ix, kgpy.vector.iy),
    ):
        with astropy.visualization.quantity_support():
            c1, c2 = components
            wire = self.wire
            if sag is not None:
                wire[z] = wire[z] + sag(wire[x], wire[y])
            if transform is not None:
                wire = transform(wire, num_extra_dims=1)
            ax.fill(wire[..., c1].T, wire[..., c2].T, fill=False)
