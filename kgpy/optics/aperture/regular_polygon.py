import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
import shapely.geometry
import kgpy.vector
from . import Polygon

__all__ = ['RegularPolygon']


@dataclasses.dataclass
class RegularPolygon(Polygon):

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
        poly = shapely.geometry.Polygon(self.wire)
        return poly.contains(p)

    @property
    def vertices(self) -> u.Quantity:
        angles = np.linspace(0, 360, self.num_sides, endpoint=False) << u.deg
        return kgpy.vector.from_components_cylindrical(self.radius, angles)
