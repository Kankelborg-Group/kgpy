import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
import kgpy.vector
from .. import material, aperture
from . import Standard

__all__ = ['Toroidal']

MaterialT = typ.TypeVar('MaterialT', bound=material.Material)
ApertureT = typ.TypeVar('ApertureT', bound=aperture.Aperture)


@dataclasses.dataclass
class Toroidal(Standard[MaterialT, ApertureT]):

    radius_of_rotation: u.Quantity = 0 * u.mm

    @property
    def __init__args(self) -> typ.Dict[str, typ.Any]:
        args = super().__init__args
        args.update({
            'radius_of_rotation': self.radius_of_rotation,
        })
        return args

    def to_zemax(self) -> 'Toroidal':
        from kgpy.optics import zemax
        return zemax.system.surface.Toroidal(**self.__init__args)

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.radius_of_rotation,
        )

    def sag(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        x2 = np.square(x)
        y2 = np.square(y)
        c = self.curvature
        r = self.radius_of_rotation
        mask = np.abs(x) > r
        zy = c * y2 / (1 + np.sqrt(1 - (1 + self.conic) * np.square(c) * y2))
        z = r - np.sqrt(np.square(r - zy) - x2)
        z[mask] = 0
        return z

    def normal(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        x2 = np.square(x)
        y2 = np.square(y)
        c = self.curvature
        c2 = np.square(c)
        r = self.radius_of_rotation
        g = np.sqrt(1 - (1 + self.conic) * c2 * y2)
        zy = c * y2 / (1 + g)
        f = np.sqrt(np.square(r - zy) - x2)
        dzdx = x / f
        dzydy = c * y / g
        dzdy = (r - zy) * dzydy / f
        return kgpy.vector.normalize(kgpy.vector.from_components(dzdx, dzdy, -1 * u.dimensionless_unscaled))
