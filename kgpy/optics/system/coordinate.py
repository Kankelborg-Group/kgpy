import dataclasses
import typing as typ
import numpy as np
import astropy.units as u

__all__ = ['Transform']

from . import mixin


@dataclasses.dataclass
class Transform(mixin.ConfigBroadcast):

    tilt: u.Quantity = dataclasses.field(default_factory=lambda: [0, 0, 0] * u.deg)
    decenter: u.Quantity = dataclasses.field(default_factory=lambda: [0, 0, 0] * u.mm)
    tilt_first: np.ndarray[bool] = dataclasses.field(default_factory=lambda: np.array(False))

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.decenter[..., 0],
            self.tilt[..., 0],
            self.tilt_first,
        )

    def __invert__(self) -> 'Transform':
        return type(self)(-self.tilt, -self.decenter, np.logical_not(self.tilt_first))

    def __add__(self, other: 'Transform'):

        if self.tilt_first != other.tilt_first:
            raise ValueError('Adding Transforms with opposite `tilt_first` attributes not supported')

        return Transform(
            tilt=self.tilt + other.tilt,
            decenter=self.decenter + other.decenter,
            tilt_first=self.tilt_first,
        )
