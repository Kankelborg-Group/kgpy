import dataclasses

import numpy as np
from astropy import units as u
from ... import mixin

__all__ = ['Decenter', 'InverseDecenter']


@dataclasses.dataclass
class Decenter(mixin.Broadcastable):
    x: u.Quantity = 0 * u.mm
    y: u.Quantity = 0 * u.mm

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
        )


@dataclasses.dataclass
class InverseDecenter:
    
    _decenter: Decenter

    @property
    def config_broadcast(self):
        return self._decenter.config_broadcast
    
    @property
    def x(self) -> u.Quantity:
        return -self._decenter.x
    
    @property
    def y(self) -> u.Quantity:
        return -self._decenter.y
