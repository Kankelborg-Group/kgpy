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
    offset_angle: u.Quantity = 0 * u.deg

    def to_zemax(self) -> 'RegularPolygon':
        raise NotImplementedError

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.radius,
        )

    @property
    def vertices(self) -> u.Quantity:
        angles = np.linspace(self.offset_angle, 360 * u.deg + self.offset_angle, self.num_sides, endpoint=False)
        return kgpy.vector.from_components_cylindrical(self.radius, angles)
