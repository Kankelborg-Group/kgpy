import typing as typ
import dataclasses
import numpy as np
from astropy import units as u
import kgpy.vector
from . import Transform

__all__ = ['Decenter']


@dataclasses.dataclass
class Decenter(Transform):

    x: u.Quantity = 0 * u.mm
    y: u.Quantity = 0 * u.mm

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.x,
            self.y,
        )

    def __invert__(self) -> 'Decenter':
        return Decenter(
            -self.x,
            -self.y,
        )

    @property
    def rotation_eff(self) -> u.Quantity:
        return super().rotation_eff

    @property
    def translation_eff(self) -> u.Quantity:
        return kgpy.vector.from_components(x=self.x, y=self.y)

    def copy(self):
        return Decenter(
            x=self.x.copy(),
            y=self.y.copy(),
        )

