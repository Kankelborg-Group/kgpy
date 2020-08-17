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
from .. import ZemaxCompatible, OCC_Compatible

__all__ = ['Aperture']


@dataclasses.dataclass
class Aperture(
    ZemaxCompatible,
    kgpy.mixin.Copyable,
    kgpy.mixin.Broadcastable,
    abc.ABC
):
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

    def global_wire(
            self,
            surface: typ.Optional['kgpy.optics.Surface'] = None,
            apply_sag: bool = True
    ):
        wire = self.wire
        if apply_sag:
            wire[kgpy.vector.z] = surface.sag(wire[kgpy.vector.x], wire[kgpy.vector.y])
        wire = surface.transform_to_global(wire, num_extra_dims=1)
        return wire

    def plot_2d(
            self,
            ax: plt.Axes,
            components: typ.Tuple[int, int] = (kgpy.vector.ix, kgpy.vector.iy),
            surface: typ.Optional['kgpy.optics.Surface'] = None,
    ):
        with astropy.visualization.quantity_support():
            c1, c2 = components
            wire = self.global_wire(surface)
            ax.fill(wire[..., c1].T, wire[..., c2].T, fill=False)
