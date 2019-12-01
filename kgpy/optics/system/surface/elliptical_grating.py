import dataclasses
import typing as tp
import numpy as np
import astropy.units as u

from . import DiffractionGrating

__all__ = ['EllipticalGrating1']


@dataclasses.dataclass
class EllipticalGrating1(DiffractionGrating):

    a: u.Quantity = dataclasses.field(default_factory=lambda: 0 / u.m)
    b: u.Quantity = dataclasses.field(default_factory=lambda: 0 / u.m)
    c: u.Quantity = dataclasses.field(default_factory=lambda: 0 * u.m)
    alpha: u.Quantity = dataclasses.field(default_factory=lambda: 0 * u.dimensionless_unscaled)
    beta: u.Quantity = dataclasses.field(default_factory=lambda: 0 * u.dimensionless_unscaled)

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.a,
            self.b,
            self.c,
            self.alpha,
            self.beta,
        )