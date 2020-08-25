import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
from kgpy import vector
from .. import material, aperture
from . import Standard

__all__ = ['Toroidal']

MaterialT = typ.TypeVar('MaterialT', bound=material.Material)
ApertureT = typ.TypeVar('ApertureT', bound=aperture.Aperture)


@dataclasses.dataclass
class Toroidal(Standard[MaterialT, ApertureT]):

    radius_of_rotation: u.Quantity = 0 * u.mm

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.radius_of_rotation,
        )

    @property
    def is_plane(self) -> np.ndarray:
        return np.isinf(self.radius) & np.isinf(self.radius_of_rotation)

    @property
    def is_sphere(self) -> np.ndarray:
        return (self.conic == 0) & (self.radius == self.radius_of_rotation)

    def sag(self, ax: u.Quantity, ay: u.Quantity) -> u.Quantity:
        x2 = np.square(ax)
        y2 = np.square(ay)
        c = self.curvature
        r = self.radius_of_rotation
        mask = np.abs(ax) > r
        zy = c * y2 / (1 + np.sqrt(1 - (1 + self.conic) * np.square(c) * y2))
        z = r - np.sqrt(np.square(r - zy) - x2)
        z[mask] = (r - np.sqrt(np.square(r - zy) - np.square(r)))[mask]
        return z

    def normal(self, ax: u.Quantity, ay: u.Quantity) -> u.Quantity:
        x2 = np.square(ax)
        y2 = np.square(ay)
        c = self.curvature
        c2 = np.square(c)
        r = self.radius_of_rotation
        g = np.sqrt(1 - (1 + self.conic) * c2 * y2)
        zy = c * y2 / (1 + g)
        f = np.sqrt(np.square(r - zy) - x2)
        dzdx = ax / f
        dzydy = c * ay / g
        dzdy = (r - zy) * dzydy / f
        mask = np.abs(ax) > r
        dzdx[mask] = 0
        dzdy[mask] = 0
        return vector.normalize(vector.from_components(dzdx, dzdy, -1 * u.dimensionless_unscaled))

    def copy(self) -> 'Toroidal':
        return Toroidal(
            name=self.name.copy(),
            thickness=self.thickness.copy(),
            is_active=self.is_active.copy(),
            is_visible=self.is_visible.copy(),
            radius=self.radius.copy(),
            conic=self.conic.copy(),
            material=self.material.copy(),
            aperture=self.aperture.copy(),
            transform_before=self.transform_before.copy(),
            transform_after=self.transform_after.copy(),
            intercept_error=self.intercept_error.copy(),
            radius_of_rotation=self.radius_of_rotation.copy(),
        )
