import dataclasses
import typing as typ
import numpy as np
import astropy.units as u

__all__ = ['Tilt', 'Decenter', 'Transform']

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


@dataclasses.dataclass
class Decenter(mixin.ConfigBroadcast):
    x: u.Quantity = 0 * u.mm
    y: u.Quantity = 0 * u.mm
    z: u.Quantity = 0 * u.mm

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
