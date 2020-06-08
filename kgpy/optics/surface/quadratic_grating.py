import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
from .. import material, aperture
from . import DiffractionGrating

__all__ = ['QuadraticGrating']

MaterialT = typ.TypeVar('MaterialT', bound=material.Material)
ApertureT = typ.TypeVar('ApertureT', bound=aperture.Aperture)


@dataclasses.dataclass
class QuadraticGrating(DiffractionGrating):

    c1: u.Quantity = dataclasses.field(default_factory=lambda: 0 * u.dimensionless_unscaled)
    c2: u.Quantity = dataclasses.field(default_factory=lambda: 0 * u.dimensionless_unscaled)

    @property
    def __init__args(self) -> typ.Dict[str, typ.Any]:
        args = super().__init__args
        args.update({
            'c1': self.c1,
            'c2': self.c2,
        })
        return args

    def to_zemax(self) -> 'QuadraticGrating':
        raise NotImplementedError

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.c1,
            self.c2,
        )

    def groove_normal(self, x: u.Quantity, y: u.Quantity):
        d = 1 / self.groove_frequency + self.c1 * y + self.c2 * np.square(y)
        return 1 / d
