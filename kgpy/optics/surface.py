
from unittest import TestCase
from numbers import Real
from typing import Union, List, Tuple
import numpy as np
import quaternion as q
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

    def __init__(self, name: str, thickness: u.Quantity = 0.0 * u.m, comment: str = '',
                 tilt: q.quaternion = q.from_euler_angles(0,0,0), decenter: Union[[Real, Real], Tuple[Real]] = (0,0)):
        """
        Constructor for the Surface class.
        This constructor places the surface at the origin of the global coordinate system, it needs to be moved into
        place after the call to this function.
        :param name: Human-readable name of the surface
        :param thickness: Thickness of the surface along the local z-direction, measured in mm
        :param comment: Additional description of this surface
        :param tilt: Rotation of the surface with respect to the coordinate system cs
        :param decenter: Translation of the surface in the directions orthogonal to the z-hat direction of coordinate
        system cs. Argument must be a two-element list or tuple [x, y]
        """

        # Check that the thickness parameter has the units attribute
        if not isinstance(thickness, u.Quantity):
            raise TypeError('thickness parameter must be an astropy.units.Quantity')

        # Check that the thickness parameter has dimensions of length
        if not thickness.unit.is_equivalent(u.m):
            raise TypeError('thickness parameter does not have dimensions of length')

        # Save input arguments as class variables
        self.name = name
        self.thickness = thickness
        self.comment = comment

        # Initialize the coordinate system to the global coordinate system
        self.cs = GlobalCoordinateSystem()
        self._tilt_dec = GlobalCoordinateSystem()

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

