import abc
import dataclasses
import typing as typ
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import kgpy.mixin
import kgpy.vector
import kgpy.optics
from .. import ZemaxCompatible

__all__ = ['Aperture']


@dataclasses.dataclass
class Aperture(ZemaxCompatible, kgpy.mixin.Broadcastable, abc.ABC):
    num_samples: int = 1000
    is_active: bool = True
    is_test_stop: bool = True

    def to_zemax(self) -> 'Aperture':
        from kgpy.optics import zemax
        return zemax.system.surface.aperture.Aperture()

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

    # @property
    # def vertices(self) -> u.Quantity:
    #     x_min = self.edges[kgpy.vector.x].min(~0, keepdims=True)
    #     x_max = self.edges[kgpy.vector.x].max(~0, keepdims=True)
    #     y_min = self.edges[kgpy.vector.y].min(~0, keepdims=True)
    #     y_max = self.edges[kgpy.vector.y].max(~0, keepdims=True)
    #
    #     zero = np.zeros_like(x_min)
    #     x = np.stack([zero, x_min, zero, x_max], axis=~0)
    #     y = np.stack([y_min, zero, y_max, zero], axis=~0)
    #     z = np.stack([zero, zero, zero, zero], axis=~0)
    #
    #     return np.stack([x, y, z], axis=~0)

    def plot_2d(
            self,
            ax: plt.Axes,
            components: typ.Tuple[int, int] = (kgpy.vector.ix, kgpy.vector.iy),
            system: typ.Optional['kgpy.optics.System'] = None,
            surface: typ.Optional['kgpy.optics.Surface'] = None,
    ):
        with astropy.visualization.quantity_support():
            c1, c2 = components
            wire = self.wire.copy()
            wire[kgpy.vector.z] = surface.sag(wire[kgpy.vector.x], wire[kgpy.vector.y])
            wire = surface.transform_to_global(wire, system, num_extra_dims=1)
            ax.fill(wire[..., c1].T, wire[..., c2].T, fill=False)
