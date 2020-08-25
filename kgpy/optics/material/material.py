import abc
import dataclasses
import typing as typ
import matplotlib.pyplot as plt
import astropy.units as u
from kgpy import mixin, vector, transform
from .. import Rays, Aperture

__all__ = ['Material']


@dataclasses.dataclass
class Material(
    mixin.Copyable,
    mixin.Broadcastable,
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
            rigid_transform: typ.Optional[transform.rigid.Transform] = None,
            components: typ.Tuple[int, int] = (vector.ix, vector.iy),
    ):
        pass
