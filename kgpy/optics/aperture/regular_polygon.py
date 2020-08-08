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

    @property
    def edge_subtent(self):
        """
        Calculate the angle subtended by each edge of the polygon by dividing the angle of a circle (360 degrees) by
        the number of sides in the regular polygon.
        :return: Angle subtended by each edge
        """
        return 360 * u.deg / self.num_sides

    @property
    def min_radius(self):
        """
        Calculate the distance from the center of the polygon to the center of an edge of a polygon.
        :return: The minimum radius of the polygon.
        """
        return self.radius * np.cos(self.edge_subtent / 2)
