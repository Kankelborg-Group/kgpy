
import numpy as np

from typing import List, Tuple
import astropy.units as u

__all__ = ['Aperture', 'Rectangular', 'Circular', 'Spider', 'Polygon', 'RegularPolygon', 'Octagon']


class Aperture:

    def __init__(self):

        self.is_obscuration = False

        self.decenter_x = 0 * u.mm      # type: u.Quantity
        self.decenter_y = 0 * u.mm      # type: u.Quantity


class Rectangular(Aperture):

    def __init__(self, half_width_x: u.Quantity = 0 * u.mm, half_width_y: u.Quantity = 0 * u.mm):

        Aperture.__init__(self)

        self.half_width_x = half_width_x
        self.half_width_y = half_width_y


class Circular(Aperture):

    def __init__(self, min_radius: u.Quantity = 0 * u.mm, max_radius: u.Quantity = 0 * u.mm):

        Aperture.__init__(self)

        self.min_radius = min_radius
        self.max_radius = max_radius


class Spider(Aperture):

    def __init__(self, arm_width: u.Quantity = 0 * u.mm, num_arms: int = 0):

        Aperture.__init__(self)

        self.num_arms = num_arms
        self.arm_width = arm_width


class Polygon(Aperture):

    def __init__(self, points: u.Quantity):

        Aperture.__init__(self)

        self.points = points


class RegularPolygon(Polygon):

    def __init__(self, radius: u.Quantity, num_sides: int):

        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False) * u.rad        # type: u.Quantity

        # Calculate points
        x = radius * np.cos(angles)      # type: u.Quantity
        y = radius * np.sin(angles)      # type: u.Quantity
        pts = np.array([x, y])
        pts = pts.transpose() * x.unit

        Polygon.__init__(self, pts)


class Octagon(RegularPolygon):

    def __init__(self, radius: u.Quantity):

        RegularPolygon.__init__(self, radius, 8)


