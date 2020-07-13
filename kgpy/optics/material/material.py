import abc
import dataclasses
import typing as typ
import matplotlib.pyplot as plt
import astropy.units as u
import kgpy.mixin
import kgpy.optics
from .. import ZemaxCompatible

__all__ = ['Material']


@dataclasses.dataclass
class Material(ZemaxCompatible, kgpy.mixin.Broadcastable, abc.ABC):

    @abc.abstractmethod
    def index_of_refraction(self, wavelength: u.Quantity, polarization: typ.Optional[u.Quantity]) -> u.Quantity:
        pass

    @property
    @abc.abstractmethod
    def propagation_signum(self) -> float:
        pass

    def plot_2d(
            self,
            ax: plt.Axes,
            components: typ.Tuple[int, int] = (kgpy.vector.ix, kgpy.vector.iy),
            system: typ.Optional['kgpy.optics.System'] = None,
            surface: typ.Optional['kgpy.optics.surface.Standard'] = None,
    ):
        pass
