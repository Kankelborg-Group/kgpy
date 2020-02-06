import dataclasses
import numpy as np

from kgpy.optics.system.coordinate.decenter import Decenter
from kgpy.optics.system.coordinate.tilt import Tilt

__all__ = ['Transform']

from kgpy.optics.system import mixin


@dataclasses.dataclass
class ZemaxTransform(mixin.ConfigBroadcast):
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

    def __invert__(self) -> 'Transform':
        return type(self)(
            self.tilt.__invert__(),
            self.decenter.__invert__(),
            not self.tilt_first,
        )


@dataclasses.dataclass
class Transform(mixin.ConfigBroadcast):
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

    def __invert__(self) -> 'Transform':
        return type(self)(
            self.tilt.__invert__(),
            self.decenter.__invert__(),
            not self.tilt_first,
        )
