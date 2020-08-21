import abc
import dataclasses
import typing as typ
import matplotlib.pyplot as plt
import astropy.units as u
import kgpy.mixin
import kgpy.vector
from .. import coordinate, Rays, Aperture

__all__ = ['Material']


@dataclasses.dataclass
class Material(
    kgpy.mixin.Copyable,
    kgpy.mixin.Broadcastable,
    abc.ABC
):

    @abc.abstractmethod
    def index_of_refraction(self, rays: Rays) -> u.Quantity:
        pass

    def plot_2d(
            self,
            ax: plt.Axes,
            aperture: Aperture,
            sag: typ.Optional[typ.Callable[[u.Quantity, u.Quantity], u.Quantity]] = None,
            transform: typ.Optional[coordinate.Transform] = None,
            components: typ.Tuple[int, int] = (kgpy.vector.ix, kgpy.vector.iy),
    ):
        pass
