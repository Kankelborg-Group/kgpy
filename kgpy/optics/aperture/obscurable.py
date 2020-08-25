import dataclasses
import numpy as np
from kgpy import mixin

__all__ = ['Obscurable']


@dataclasses.dataclass
class Obscurable(
    mixin.Broadcastable,
    mixin.Copyable,
):

    is_obscuration: bool = False

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.is_obscuration,
        )

    def copy(self) -> 'Obscurable':
        other = super().copy()      # type: Obscurable
        other.is_obscuration = self.is_obscuration
        return other
