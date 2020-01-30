import dataclasses
import typing as typ
import numpy as np
import astropy.units as u

from . import Standard

__all__ = ['Toroidal']

AperSurfT = typ.TypeVar('AperSurfT')
MainSurfT = typ.TypeVar('MainSurfT')


@dataclasses.dataclass
class Toroidal(Standard[AperSurfT, MainSurfT]):

    radius_of_rotation: u.Quantity = 0 * u.mm

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.radius_of_rotation,
        )