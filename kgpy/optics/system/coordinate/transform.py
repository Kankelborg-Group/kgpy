import dataclasses
import numpy as np

from kgpy.optics.system import mixin
from . import Tilt, Translate


@dataclasses.dataclass
class Transform(mixin.ConfigBroadcast):
    tilt: Tilt = dataclasses.field(default_factory=lambda: Tilt())
    translate: Translate = dataclasses.field(default_factory=lambda: Translate())
    tilt_first: bool = False

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.tilt.config_broadcast,
            self.translate.config_broadcast,
        )

    def __invert__(self) -> 'Transform':
        return type(self)(
            self.tilt.__invert__(),
            self.translate.__invert__(),
            not self.tilt_first,
        )
