import abc
import dataclasses
import typing as typ
import matplotlib.pyplot as plt
import astropy.units as u
import kgpy.mixin
import kgpy.optics
from .. import ZemaxCompatible, Rays

__all__ = ['Material']


@dataclasses.dataclass
class Material(
    ZemaxCompatible,
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
            components: typ.Tuple[int, int] = (kgpy.vector.ix, kgpy.vector.iy),
            surface: typ.Optional['kgpy.optics.surface.Standard'] = None,
    ):
        pass
