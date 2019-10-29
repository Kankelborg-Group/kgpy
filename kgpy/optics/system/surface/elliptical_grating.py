import dataclasses
import typing as tp
import astropy.units as u

from . import DiffractionGrating

__all__ = ['EllipticalGrating1']


@dataclasses.dataclass
class EllipticalGrating1(DiffractionGrating):

    a: u.Quantity = 0 * u.dimensionless_unscaled
    b: u.Quantity = 0 * u.dimensionless_unscaled
    c: u.Quantity = 0 * u.m
    alpha: u.Quantity = 0 * u.dimensionless_unscaled
    beta: u.Quantity = 0 * u.dimensionless_unscaled

    @property
    def broadcastable_attrs(self):
        return super().broadcastable_attrs + [
            self.a,
            self.b,
            self.c,
            self.alpha,
            self.beta,
        ]