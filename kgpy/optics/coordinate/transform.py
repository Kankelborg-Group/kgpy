import dataclasses
import numpy as np
import astropy.units as u
import kgpy.mixin
from . import Tilt, Translate, TiltDecenter

__all__ = ['Transform']


@dataclasses.dataclass
class Transform(kgpy.mixin.Broadcastable):
    tilt: Tilt = dataclasses.field(default_factory=lambda: Tilt())
    translate: Translate = dataclasses.field(default_factory=lambda: Translate())
    tilt_first: bool = False

    @classmethod
    def from_tilt_decenter(
            cls,
            tilt_decenter: TiltDecenter = None,
            z=0 * u.mm,
    ):
        return cls(
            tilt=tilt_decenter.tilt.copy(),
            translate=Translate(
                x=tilt_decenter.decenter.x.copy(),
                y=tilt_decenter.decenter.y.copy(),
                z=z,
            )
        )

    @classmethod
    def promote(cls, value: 'Transform'):
        return cls(value.tilt, value.translate, value.tilt_first)

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.tilt.config_broadcast,
            self.translate.config_broadcast,
        )

    def __invert__(self):
        return type(self)(
            self.tilt.__invert__(),
            self.translate.__invert__(),
            not self.tilt_first,
        )

    def __call__(
            self,
            value: u.Quantity,
            tilt: bool = True,
            translate: bool = True,
            inverse: bool = False,
            num_extra_dims: int = 0,
    ) -> u.Quantity:
        value = value.copy()
        if not self.tilt_first:
            if translate:
                value = self.translate(value, inverse, num_extra_dims)
            if tilt:
                value = self.tilt(value, inverse, num_extra_dims)
        else:
            if tilt:
                value = self.tilt(value, inverse, num_extra_dims)
            if translate:
                value = self.translate(value, inverse, num_extra_dims)
        return value

    def copy(self) -> 'Transform':
        return type(self)(
            tilt=self.tilt.copy(),
            translate=self.translate.copy(),
            tilt_first=self.tilt_first,
        )

