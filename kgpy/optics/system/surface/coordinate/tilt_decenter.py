import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
from ... import mixin
from . import Decenter, Tilt

__all__ = ['TiltDecenter']


@dataclasses.dataclass
class TiltDecenter(mixin.InitArgs, mixin.Broadcastable):
    tilt: Tilt = dataclasses.field(default_factory=lambda: Tilt())
    decenter: Decenter = dataclasses.field(default_factory=lambda: Decenter())
    tilt_first: bool = False

    @classmethod
    def promote(cls, value: 'TiltDecenter'):
        return cls(value.tilt, value.decenter, value.tilt_first)

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.tilt.config_broadcast,
            self.decenter.config_broadcast,
        )

    @property
    def __init__args(self) -> typ.Dict[str, typ.Any]:
        args = super().__init__args
        args.update({
            'tilt': self.tilt_first,
            'decenter': self.decenter,
            'tilt_first': self.tilt_first,
        })
        return args

    def __invert__(self):
        return type(self)(
            self.tilt.__invert__(),
            self.decenter.__invert__(),
            self.tilt_first.__invert__(),
        )

    def apply(
            self,
            value: u.Quantity,
            tilt: bool = True,
            decenter: bool = True,
            inverse: bool = False,
    ) -> u.Quantity:
        value = value.copy()
        if not self.tilt_first:
            if decenter:
                value = self.decenter.apply(value, inverse)
            if tilt:
                value = self.tilt.apply(value, inverse)
        else:
            if tilt:
                value = self.tilt.apply(value, inverse)
            if decenter:
                value = self.decenter.apply(value, inverse)
        return value
