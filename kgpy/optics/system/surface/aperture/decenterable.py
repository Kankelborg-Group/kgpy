import dataclasses
import numpy as np

from ... import mixin
from .. import coordinate

__all__ = ['Decenterable']


@dataclasses.dataclass
class Decenterable(mixin.Broadcastable):

    decenter: coordinate.Decenter = dataclasses.field(default_factory=lambda: coordinate.Decenter())

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.decenter,
        )
