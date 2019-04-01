
from typing import List, Tuple
from shapely.geometry import Polygon
import astropy.units as u

from kgpy.math import Vector

__all__ = ['Aperture', 'Rectangular', 'Circular', 'Spider', 'User']


class Aperture:

    def __init__(self):

        self.is_obscuration = False

        self.decenter_x = 0 * u.mm      # type: u.Quantity
        self.decenter_y = 0 * u.mm      # type: u.Quantity


class Rectangular(Aperture):

    def __init__(self, half_width_x: u.Quantity, half_width_y: u.Quantity):

        super().__init__()

        self.half_width_x = half_width_x
        self.half_width_y = half_width_y


class Circular(Aperture):

    def __init__(self, min_radius: u.Quantity, max_radius: u.Quantity):

        super().__init__()

        self.min_radius = min_radius
        self.max_radius = max_radius


class Spider(Aperture):

    def __init__(self, arm_width: u.Quantity, num_arms: int):

        super().__init__()

        self.num_arms = num_arms
        self.arm_width = arm_width


class User(Aperture):

    def __init__(self, points: List[Tuple[u.Quantity, u.Quantity]]):

        super().__init__()

        self.points = points
