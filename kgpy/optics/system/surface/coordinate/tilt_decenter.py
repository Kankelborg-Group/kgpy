import dataclasses
import typing as typ
import numpy as np
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
