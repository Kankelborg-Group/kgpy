import dataclasses
import numpy as np
from ... import mixin
from . import Tilt, Translate

__all__ = ['Transform']


@dataclasses.dataclass
class Transform(mixin.Broadcastable):
    tilt: Tilt = dataclasses.field(default_factory=lambda: Tilt())
    translate: Translate = dataclasses.field(default_factory=lambda: Translate())
    tilt_first: bool = False

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


