import dataclasses
import typing as typ
import numpy as np
import astropy.units as u

from kgpy.optics.system.surface.aperture import Aperture

__all__ = ['RegularPolygon']


@dataclasses.dataclass
class RegularPolygon(Aperture):

    radius: u.Quantity = 0 * u.mm
    num_sides: int = 0

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.radius,
            self.num_sides,
        )

    @property
    def points(self) -> u.Quantity:
        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, self.num_sides, endpoint=False) * u.rad  # type: u.Quantity

        # Calculate points
        x = self.radius * np.cos(angles)  # type: u.Quantity
        y = self.radius * np.sin(angles)  # type: u.Quantity
        pts = u.Quantity([x, y])
        pts = u.Quantity(pts.transpose())
        pts = u.Quantity([pts])

        return pts
