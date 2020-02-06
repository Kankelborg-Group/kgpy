import dataclasses

import numpy as np
from astropy import units as u

from kgpy.optics.system import mixin


@dataclasses.dataclass
class Decenter(mixin.ConfigBroadcast):
    x: u.Quantity = 0 * u.mm
    y: u.Quantity = 0 * u.mm
    y_first: bool = False

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.x,
            self.y,
        )

    def __invert__(self):
        return type(self)(
            -self.x,
            -self.y,
            not self.y_first,
        )
