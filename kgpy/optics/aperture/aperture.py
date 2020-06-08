import abc
import dataclasses
import typing as typ
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import kgpy.mixin
import kgpy.vector
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
    def edges(self) -> typ.Optional[u.Quantity]:
        pass

    @abc.abstractmethod
    def plot_2d(
            self,
            ax: plt.Axes,
            components: typ.Tuple[int, int] = (kgpy.vector.ix, kgpy.vector.iy),
            transform_func: typ.Optional[typ.Callable[[u.Quantity, bool], u.Quantity]] = None,
            sag_func: typ.Optional[typ.Callable[[u.Quantity, u.Quantity], u.Quantity]] = None,
    ):
        edges = self.edges.copy()
        edges[kgpy.vector.z] = sag_func(edges[kgpy.vector.x], edges[kgpy.vector.y])
        edges = transform_func(edges)




