import dataclasses
import numpy as np
from kgpy import mixin, transform

__all__ = ['Decenterable']


@dataclasses.dataclass
class Decenterable(mixin.Broadcastable):

    decenter: transform.rigid.Translate = dataclasses.field(default_factory=lambda: transform.rigid.Translate())

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.decenter,
        )
