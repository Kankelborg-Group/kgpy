
from unittest import TestCase
import numpy as np
import quaternion
import astropy.units as u
from astropy.coordinates import Distance

from kgpy.math import CoordinateSystem, Vector
from kgpy.math.coordinate_system import GlobalCoordinateSystem

__all__ = ['Surface']


class Surface:
    """
    This class represents a single optical surface. This class should be a drop-in replacement for a Zemax surface, and
    have all the same properties and behaviors.
    """

    def __init__(self, name: str, thickness: u.Quantity = 0.0 * u.m, comment: str = ''):
        """
        Constructor for the Surface class.
        This constructor places the surface at the origin of the global coordinate system, it needs to be moved into
        place after the call to this function.
        :param name: Human-readable name of the surface
        :param thickness: Thickness of the surface along the local z-direction, measured in mm
        :param comment: Additional description of this surface
        """

        # Check that the thickness parameter has dimensions of length
        if not thickness.unit.is_equivalent(u.m):
            raise TypeError('thickness parameter does not have dimensions of length')

        # Save input arguments as class variables
        self.name = name
        self.thickness = thickness
        self.comment = comment

        # Initialize the coordinate system to the global coordinate system
        self.cs = GlobalCoordinateSystem()

    @property
    def T(self) -> Vector:
        """
        Thickness vector
        :return: Vector pointing from the center of a surface's front face to the center of a surface's back face
        """
        return self.thickness * self.cs.zh

    def __str__(self) -> str:
        """
        :return: String representation of the surface
        """

        return self.name + ', comment = ' + self.comment + ', thickness = ' + str(self.thickness) + ', cs = [' + self.cs.__str__() + ']'

