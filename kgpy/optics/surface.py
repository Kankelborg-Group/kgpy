
from unittest import TestCase
from numbers import Real
from typing import Union, List, Tuple
import numpy as np
import quaternion as q
import astropy.units as u
from astropy.coordinates import Distance


from kgpy.math import CoordinateSystem, Vector
from kgpy.math.coordinate_system import GlobalCoordinateSystem as gcs

__all__ = ['Surface']


class Surface:
    """
    This class represents a single optical surface. This class should be a drop-in replacement for a Zemax surface, and
    have all the same properties and behaviors.
    """

    def __init__(self, name: str, thickness: u.Quantity = 0.0 * u.m, comment: str = '',
                 tilt_dec: CoordinateSystem = gcs(), cs_break: CoordinateSystem = gcs()):
        """
        Constructor for the Surface class.
        This constructor places the surface at the origin of the global coordinate system, it needs to be moved into
        place after the call to this function.
        :param name: Human-readable name of the surface
        :param thickness: Thickness of the surface along the local z-direction, measured in mm
        :param comment: Additional description of this surface
        :param tilt_dec: Temporary coordinate system break to place the surface, i.e. the coordinate system after this
        surface ignores this argument.
        :param cs_break: Permanent coordinate system break at the front face of this surface, i.e. the coordinate system
        after this surface incorporates this argument
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
        self.tilt_dec = tilt_dec
        self.cs_break = cs_break

        # Attributes to be set by Component.append_surface()
        # These are links to the previous surface in the component and the overall component itself.
        # These are used to recursively calculate properties of this surface instead of explicitly updating
        # this surface.
        # For example if the thickness of an earlier surface changes, we would have to remember to update this surface
        # with a new global position.
        # However if we calculate the global position by adding up the thickness of all previous surfaces, it is not
        # necessary to remember to update anything.
        self.previous_surf = None   # type: 'Surface'
        self.component = None       # type 'Component'


    @property
    def T(self) -> Vector:
        """
        Thickness vector
        :return: Vector pointing from the center of a surface's front face to the center of a surface's back face
        """
        return self.thickness * self.cs.zh

    @property
    def front_cs(self) -> CoordinateSystem:

         return self.component.cs @ (self.previous_surf.back_cs @ (self.cs_break @ self.tilt_dec) )

    @property
    def back_cs(self) -> CoordinateSystem:

        pass

    def __str__(self) -> str:
        """
        :return: String representation of the surface
        """

        return self.name + ', comment = ' + self.comment + ', thickness = ' + str(self.thickness) + ', cs = [' \
               + self.cs.__str__() + ']'

