import dataclasses
import numpy as np
import astropy.units as u
from kgpy import vector, optimization
from .. import Rays
from . import Sag

__all__ = ['Standard']


@dataclasses.dataclass
class Standard(Sag):
    radius: u.Quantity = np.inf * u.mm
    conic: u.Quantity = 0 * u.dimensionless_unscaled

    @property
    def curvature(self) -> u.Quantity:
        return np.where(np.isinf(self.radius), 0 / u.mm, 1 / self.radius)

    def __call__(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        r2 = np.square(x) + np.square(y)
        c = self.curvature
        sz = c * r2 / (1 + np.sqrt(1 - (1 + self.conic) * np.square(c) * r2))
        mask = r2 >= np.square(self.radius)
        sz[mask] = 0
        return sz

    def normal(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        x2, y2 = np.square(x), np.square(y)
        c = self.curvature
        c2 = np.square(c)
        g = np.sqrt(1 - (1 + self.conic) * c2 * (x2 + y2))
        dzdx, dzdy = c * x / g, c * y / g
        mask = (x2 + y2) >= np.square(self.radius)
        dzdx[mask] = 0
        dzdy[mask] = 0
        n = vector.normalize(vector.from_components(dzdx, dzdy, -1 * u.dimensionless_unscaled))
        return n

    def copy(self) -> 'Standard':
        other = super().copy()  # type: Standard
        other.radius = self.radius.copy()
        other.conic = self.conic.copy()

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.radius)
        out = np.broadcast(out, self.conic)
        return out
