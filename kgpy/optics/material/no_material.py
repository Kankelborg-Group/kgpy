import dataclasses
import typing as typ
import matplotlib.pyplot as plt
from astropy import units as u
import kgpy.vector
import kgpy.optics
from . import Material

__all__ = ['NoMaterial']


@dataclasses.dataclass
class NoMaterial(Material):

    def to_zemax(self) -> 'NoMaterial':
        from kgpy.optics import zemax
        return zemax.system.surface.material.NoMaterial()

    def index_of_refraction(self, wavelength: u.Quantity, polarization: typ.Optional[u.Quantity]):
        return 1 * u.dimensionless_unscaled

    @property
    def propagation_signum(self) -> float:
        return 1.

    def plot_2d(
            self,
            ax: plt.Axes,
            components: typ.Tuple[int, int] = (kgpy.vector.ix, kgpy.vector.iy),
            system: typ.Optional['kgpy.optics.System'] = None,
            surface: typ.Optional['kgpy.optics.surface.Standard'] = None,
    ):
        pass
