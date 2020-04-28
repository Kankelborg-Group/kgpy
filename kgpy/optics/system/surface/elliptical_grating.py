import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
from . import DiffractionGrating, material, aperture

__all__ = ['EllipticalGrating1']

MaterialT = typ.TypeVar('MaterialT', bound=material.Material)
ApertureT = typ.TypeVar('ApertureT', bound=aperture.Aperture)


@dataclasses.dataclass
class EllipticalGrating1(DiffractionGrating):

    a: u.Quantity = dataclasses.field(default_factory=lambda: 0 / u.m)
    b: u.Quantity = dataclasses.field(default_factory=lambda: 0 / u.m)
    c: u.Quantity = dataclasses.field(default_factory=lambda: 0 * u.m)
    alpha: u.Quantity = dataclasses.field(default_factory=lambda: 0 * u.dimensionless_unscaled)
    beta: u.Quantity = dataclasses.field(default_factory=lambda: 0 * u.dimensionless_unscaled)

    @property
    def __init__args(self) -> typ.Dict[str, typ.Any]:
        args = super().__init__args
        args.update({
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'alpha': self.alpha,
            'beta': self.beta,
        })

    def to_zemax(self) -> 'EllipticalGrating1':
        from kgpy.optics import zemax
        return zemax.system.surface.EllipticalGrating1(**self.__init__args)

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