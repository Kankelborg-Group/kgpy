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
    num_samples: int = 100

    def to_zemax(self) -> 'Aperture':
        from kgpy.optics import zemax
        return zemax.system.surface.aperture.Aperture()

    @abc.abstractmethod
    def is_unvignetted(self, points: u.Quantity) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def limits_x(self) -> typ.Tuple[u.Quantity, u.Quantity]:
        pass

    @property
    @abc.abstractmethod
    def limits_y(self) -> typ.Tuple[u.Quantity, u.Quantity]:
        pass

    @property
    @abc.abstractmethod
    def edges(self) -> u.Quantity:
        pass

    @property
    def vertices(self) -> u.Quantity:
        x_min = self.edges[kgpy.vector.x].min(~0, keepdims=True)
        x_max = self.edges[kgpy.vector.x].max(~0, keepdims=True)
        y_min = self.edges[kgpy.vector.y].min(~0, keepdims=True)
        y_max = self.edges[kgpy.vector.y].max(~0, keepdims=True)

        zero = np.zeros_like(x_min)
        x = np.stack([zero, x_min, zero, x_max], axis=~0)
        y = np.stack([y_min, zero, y_max, zero], axis=~0)
        z = np.stack([zero, zero, zero, zero], axis=~0)

        return np.stack([x, y, z], axis=~0)

    def plot_2d(
            self,
            ax: plt.Axes,
            components: typ.Tuple[int, int] = (kgpy.vector.ix, kgpy.vector.iy),
            system: typ.Optional['kgpy.optics.System'] = None,
            surface: typ.Optional['kgpy.optics.Surface'] = None,
    ):
        with astropy.visualization.quantity_support():
            c1, c2 = components
            edges = self.edges.copy()
            edges[kgpy.vector.z] = surface.sag(edges[kgpy.vector.x], edges[kgpy.vector.y])
            edges = surface.transform_to_global(edges, system, num_extra_dims=2)
            edges = edges.reshape(edges.shape[:~2] + (edges.shape[~2] * edges.shape[~1], edges.shape[~0]))
            ax.fill(edges[..., c1].T, edges[..., c2].T, fill=False)
