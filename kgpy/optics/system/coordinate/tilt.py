import dataclasses

import numpy as np
from astropy import units as u

from kgpy.optics.system import mixin

__all__ = ['Tilt', 'InverseTilt']


@dataclasses.dataclass
class Tilt(mixin.ConfigBroadcast):
    x: u.Quantity = 0 * u.deg
    y: u.Quantity = 0 * u.deg
    z: u.Quantity = 0 * u.deg

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.x,
            self.y,
            self.z,
        )

    def __invert__(self):
        return type(self)(
            -self.x,
            -self.y,
            -self.z,
        )


@dataclasses.dataclass
class InverseTilt:

    _tilt: Tilt

    @property
    def x(self) -> u.Quantity:
        return -self.x

    @property
    def y(self) -> u.Quantity:
        return -self.y

    @property
    def z(self) -> u.Quantity:
        return -self.z
