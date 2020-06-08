import dataclasses
import numpy as np
import kgpy.mixin

__all__ = ['Obscurable']


@dataclasses.dataclass
class Obscurable(kgpy.mixin.Broadcastable):

    is_obscuration: bool = False

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.is_obscuration,
        )
