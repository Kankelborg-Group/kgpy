import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
import shapely.geometry
import kgpy.vector
from . import Aperture, decenterable, obscurable

__all__ = ['RegularPolygon']


@dataclasses.dataclass
class RegularPolygon(decenterable.Decenterable, obscurable.Obscurable, Aperture):

    radius: u.Quantity = 0 * u.mm
    num_sides: int = 8

    def to_zemax(self) -> 'RegularPolygon':
        raise NotImplementedError

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.radius,
        )

    def is_unvignetted(self, points: u.Quantity) -> np.ndarray:
        p = shapely.geometry.Point(points[0:2])
        poly = shapely.geometry.Polygon(self.edges)
        return poly.contains(p)

    @property
    def edges(self) -> u.Quantity:

        angles = np.linspace(0, 360 * u.deg, self.num_sides, endpoint=False)
        x = self.radius * np.cos(angles)
        y = self.radius * np.sin(angles)

        return kgpy.vector.from_components(x, y, 0 * u.dimensionless_unscaled)
