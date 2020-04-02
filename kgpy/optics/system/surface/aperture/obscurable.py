import dataclasses
import numpy as np

from ... import mixin

__all__ = ['Obscurable']


@dataclasses.dataclass
class Obscurable(mixin.Broadcastable):

    is_obscuration: 'np.ndarray[bool]' = dataclasses.field(default_factory=lambda: np.array(False))

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.is_obscuration,
        )



