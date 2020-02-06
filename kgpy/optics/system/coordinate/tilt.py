import dataclasses

import numpy as np
from astropy import units as u

from kgpy.optics.system import mixin


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