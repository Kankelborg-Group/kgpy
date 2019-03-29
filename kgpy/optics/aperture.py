
from typing import List, Tuple
from shapely.geometry import Polygon
import astropy.units as u

from kgpy.math import Vector

__all__ = ['Rectangular', 'Circular', 'Spider', 'User']


class Aperture:

    def __init__(self, decenter: Vector, is_obscuration=False):

        self.is_obscuration = is_obscuration

        self.decenter = decenter

    @property
    def decenter(self) -> Vector:
        return self._decenter

    @decenter.setter
    def decenter(self, val: Vector) -> None:

        # Check that the value has units of length
        if val.X.unit.is_equivalent(u.m):

            # Check that the value has no z-component
            if val.z == 0:
                self._decenter = val

            # Otherwise, there is a z-component and this is not allowed
            else:
                raise ValueError('Decenter must not have a z-component')

        # Otherwise, the units are incorrect
        else:
            raise ValueError('Decenter must have units of length')


class Rectangular(Aperture):

    def __init__(self, half_width_x: u.Quantity, half_width_y: u.Quantity, decenter: Vector, is_obscuration=False):

        super().__init__(decenter, is_obscuration)

        self.half_width_x = half_width_x
        self.half_width_y = half_width_y

    @property
    def half_width_x(self) -> u.Quantity:
        return self._half_width_x

    @half_width_x.setter
    def half_width_x(self, val: u.Quantity) -> None:

        # Check that the value has units of length
        if val.unit.is_equivalent(u.m):
            self._half_width_x = val

        # Otherwise, the units are incorrect
        else:
            raise ValueError('Must have units of length')

    @property
    def half_width_y(self) -> u.Quantity:
        return self._half_width_y

    @half_width_y.setter
    def half_width_y(self, val: u.Quantity) -> None:

        # Check that the value has units of length
        if val.unit.is_equivalent(u.m):
            self._half_width_y = val

        # Otherwise, the units are incorrect
        else:
            raise ValueError('Must have units of length')


class Circular(Aperture):

    def __init__(self, min_radius: u.Quantity, max_radius: u.Quantity, decenter=Vector([0, 0, 0] * u.mm),
                 is_obscuration=False):

        super().__init__(decenter, is_obscuration)

        self.min_radius = min_radius
        self.max_radius = max_radius

    @property
    def min_radius(self) -> u.Quantity:
        return self._min_radius

    @min_radius.setter
    def min_radius(self, val: u.Quantity) -> None:

        # Check that the value has units of length
        if val.unit.is_equivalent(u.m):
            self._min_radius = val

        # Otherwise, the units are incorrect
        else:
            raise ValueError('Must have units of length')

    @property
    def max_radius(self) -> u.Quantity:
        return self._max_radius

    @max_radius.setter
    def max_radius(self, val: u.Quantity) -> None:

        # Check that the value has units of length
        if val.unit.is_equivalent(u.m):
            self._max_radius = val

        # Otherwise, the units are incorrect
        else:
            raise ValueError('Must have units of length')


class Spider(Aperture):

    def __init__(self, arm_width: u.Quantity, num_arms: int, decenter=Vector([0, 0, 0] * u.mm), ):

        super().__init__(decenter, is_obscuration=False)

        if arm_width.ndim != 1:
            raise ValueError('arm_width should be a 1D array')

        if arm_width.shape[0] != num_arms and arm_width.shape[0] != 1:
            raise ValueError('arm_width should be either a scalar or have the same number of elements as num_arms')

        self.arm_width = arm_width
        self.num_arms = num_arms

    @property
    def arm_width(self) -> u.Quantity:
        return self._arm_width

    @arm_width.setter
    def arm_width(self, val: u.Quantity) -> None:

        # Check that the value has units of length
        if val.unit.is_equivalent(u.m):
            self._arm_width = val

        # Otherwise, the units are incorrect
        else:
            raise ValueError('Must have units of length')


class User(Aperture):

    def __init__(self, points: List[Tuple[u.Quantity, u.Quantity]], decenter=Vector([0, 0, 0] * u.mm),
                 is_obscuration=False):

        super().__init__(decenter, is_obscuration)

        self.points = points
