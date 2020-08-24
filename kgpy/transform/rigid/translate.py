import dataclasses
import numpy as np
from astropy import units as u
import kgpy.vector
from . import Transform

__all__ = ['Translate']


@dataclasses.dataclass
class Translate(Transform):

    vector: u.Quantity = dataclasses.field(default_factory=kgpy.vector.from_components)

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.vector,
        )

    def __invert__(self) -> 'Translate':
        return Translate(-self.vector)

    @property
    def rotation_eff(self) -> None:
        return None

    @property
    def translation_eff(self) -> u.Quantity:
        return self.vector

    def copy(self) -> 'Translate':
        return Translate(
            vector=self.vector.copy()
        )


