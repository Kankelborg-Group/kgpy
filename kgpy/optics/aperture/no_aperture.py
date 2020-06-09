import typing as typ
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import kgpy.vector
import kgpy.optics
from . import Aperture

__all__ = ['NoAperture']


class NoAperture(Aperture):

    def to_zemax(self) -> 'NoAperture':
        raise NotImplementedError

    def is_unvignetted(self, points: u.Quantity) -> np.ndarray:
        return np.array(True)

    @property
    def edges(self) -> typ.Optional[u.Quantity]:
        return None

    def plot_2d(
            self,
            ax: plt.Axes,
            components: typ.Tuple[int, int] = (kgpy.vector.ix, kgpy.vector.iy),
            system: typ.Optional['kgpy.optics.System'] = None,
            surface: typ.Optional['kgpy.optics.Surface'] = None,
    ):
        pass

