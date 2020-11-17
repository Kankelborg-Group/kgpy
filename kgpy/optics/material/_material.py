import abc
import dataclasses
import typing as typ
import matplotlib.pyplot as plt
import astropy.units as u
from kgpy import mixin, vector, transform
from .. import Rays
from ..aperture import Aperture

__all__ = ['Material']


@dataclasses.dataclass
class Material(
    mixin.Broadcastable,
    mixin.Copyable,
    abc.ABC
):

    @abc.abstractmethod
    def index_of_refraction(self, rays: Rays) -> u.Quantity:
        pass

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[int, int] = (vector.ix, vector.iy),
            rigid_transform: typ.Optional[transform.rigid.TransformList] = None,
            sag: typ.Optional[typ.Callable[[u.Quantity, u.Quantity], u.Quantity]] = None,
            aperture: typ.Optional[Aperture] = None,
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        return ax
