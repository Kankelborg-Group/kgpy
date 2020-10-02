import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
from kgpy import vector
from . import ConstantDensity

__all__ = ['CubicPolyDensity']


@dataclasses.dataclass
class CubicPolyDensity(ConstantDensity):
    ruling_density_linear: u.Quantity = 0 / (u.mm ** 2)
    ruling_density_quadratic: u.Quantity = 0 / (u.mm ** 3)
    ruling_density_cubic: u.Quantity = 0 / (u.mm ** 4)

    def normal(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        x2 = np.square(x)
        term0 = self.ruling_density[..., None, None, None, None, None]
        term1 = self.ruling_density_linear * x
        term2 = self.ruling_density_quadratic * x2
        term3 = self.ruling_density_cubic * x * x2
        groove_density = term0 + term1 + term2 + term3
        return vector.from_components(x=groove_density)

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.ruling_density_linear)
        out = np.broadcast(out, self.ruling_density_quadratic)
        out = np.broadcast(out, self.ruling_density_cubic)
        return out

    def copy(self) -> 'CubicPolyDensity':
        other = super().copy()  # type: CubicPolyDensity
        other.ruling_density_linear = self.ruling_density_linear.copy()
        other.ruling_density_quadratic = self.ruling_density_quadratic.copy()
        other.ruling_density_cubic = self.ruling_density_cubic.copy()
        return other
