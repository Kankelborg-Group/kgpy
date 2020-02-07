import dataclasses

import numpy as np

from .. import mixin
from . import Decenter, Tilt


@dataclasses.dataclass
class Base(mixin.ConfigBroadcast):
    tilt: Tilt = dataclasses.field(default_factory=lambda: Tilt())
    decenter: Decenter = dataclasses.field(default_factory=lambda: Decenter())
    tilt_first: bool = False

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
            not self.tilt_first,
        )


class TiltDecenter(Base):
    
    @property
    def tilt_first(self) -> bool:
        return self._tilt_first

    @tilt_first.setter
    def tilt_first(self, value: bool):
        self._tilt_first = value
        self.tilt.z_first = value
