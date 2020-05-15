import dataclasses
import numpy as np
import kgpy.mixin
import kgpy.coordinate

__all__ = ['Decenterable']


@dataclasses.dataclass
class Decenterable(kgpy.mixin.Broadcastable):

    decenter: kgpy.coordinate.Decenter = dataclasses.field(default_factory=lambda: kgpy.coordinate.Decenter())

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.decenter,
        )
