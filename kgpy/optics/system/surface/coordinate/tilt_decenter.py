import dataclasses

import numpy as np
from ... import mixin
from . import Decenter, InverseDecenter, Tilt, InverseTilt, TiltFirst, InverseTiltFirst

__all__ = ['TiltDecenter', 'InverseTiltDecenter']


@dataclasses.dataclass
class Base:
    tilt: Tilt = dataclasses.field(default_factory=lambda: Tilt())
    decenter: Decenter = dataclasses.field(default_factory=lambda: Decenter())
    tilt_first: bool = False


@dataclasses.dataclass
class TiltDecenter(Base, mixin.Broadcastable):

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.tilt.config_broadcast,
            self.decenter.config_broadcast,
        )

    def __invert__(self):
        return type(self)(
            self.tilt.__invert__(),
            self.decenter.__invert__(),
            self.tilt_first.__invert__(),
        )
        
        
@dataclasses.dataclass
class InverseTiltDecenter:
    
    _tilt_decenter: TiltDecenter
    
    @property
    def config_broadcast(self):
        return self._tilt_decenter.config_broadcast
    
    @property
    def tilt(self) -> InverseTilt:
        return InverseTilt(self._tilt_decenter.tilt)
    
    @property
    def decenter(self) -> InverseDecenter:
        return InverseDecenter(self._tilt_decenter.decenter)
    
    @property
    def tilt_first(self) -> InverseTiltFirst:
        return InverseTiltFirst(self._tilt_decenter.tilt_first)
