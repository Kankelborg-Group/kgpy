import dataclasses
import numpy as np
from kgpy import mixin, transform

__all__ = ['Decenterable']


@dataclasses.dataclass
class Decenterable(
    mixin.Broadcastable,
    mixin.Copyable,
):

    decenter: transform.rigid.Translate = dataclasses.field(default_factory=lambda: transform.rigid.Translate())

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.decenter,
        )

    def copy(self) -> 'Decenterable':
        other = super().copy()
        other.decenter = self.decenter.copy()
        return other
