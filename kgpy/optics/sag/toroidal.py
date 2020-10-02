import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
from kgpy import vector
from . import Standard

__all__ = ['Toroidal']


@dataclasses.dataclass
class Toroidal(Standard):
    radius_of_rotation: u.Quantity = 0 * u.mm

    def __call__(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        x2 = np.square(x)
        y2 = np.square(y)
        c = self.curvature[..., None, None, None, None, None]
        r = self.radius_of_rotation[..., None, None, None, None, None]
        mask = np.abs(x) > r
        conic = self.conic[..., None, None, None, None, None]
        zy = c * y2 / (1 + np.sqrt(1 - (1 + conic) * np.square(c) * y2))
        z = r - np.sqrt(np.square(r - zy) - x2)
        z[mask] = (r - np.sqrt(np.square(r - zy) - np.square(r)))[mask]
        return z

    def normal(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        x2 = np.square(x)
        y2 = np.square(y)
        c = self.curvature[..., None, None, None, None, None]
        c2 = np.square(c)
        r = self.radius_of_rotation[..., None, None, None, None, None]
        conic = self.conic[..., None, None, None, None, None]
        g = np.sqrt(1 - (1 + conic) * c2 * y2)
        zy = c * y2 / (1 + g)
        f = np.sqrt(np.square(r - zy) - x2)
        dzdx = x / f
        dzydy = c * y / g
        dzdy = (r - zy) * dzydy / f
        mask = np.abs(x) > r
        dzdx[mask] = 0
        dzdy[mask] = 0
        return vector.normalize(vector.from_components(dzdx, dzdy, -1 * u.dimensionless_unscaled))

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.radius_of_rotation)
        return out

    def copy(self) -> 'Toroidal':
        other = super().copy()  # type: Toroidal
        other.radius_of_rotation = self.radius_of_rotation.copy()
        return other
