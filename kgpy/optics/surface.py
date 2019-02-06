
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
                 cs_break: CoordinateSystem = gcs(), tilt_dec: CoordinateSystem = gcs()):
        """
        Constructor for the Surface class.
        This constructor places the surface at the origin of the global coordinate system, it needs to be moved into
        place after the call to this function.
        :param name: Human-readable name of the surface
        :param thickness: Thickness of the surface along the self.cs.zh direction. Must have dimensions of length.
        :param comment: Additional description of this surface
        :param cs_break: CoordinateSystem applied to the surface the modifies the current CoordinateSystem.
        The main use of this argument is to change the direction of propagation for the beam.
        This argument is similar to the Coordinate Break surface in Zemax.
        In this implementation a Surface can have a coordinate break instead of needing to define a second surface.
        :param tilt_dec: CoordinateSystem applied only to the front face of the surface that leaves the current
        CoordinateSystem unchanged.
        The main use of this argument is to decenter/offset an optic but leave the direction of propagation unchanged.
        This argument is similar to the tilt/decenter feature in Zemax.
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
    def cs(self) -> CoordinateSystem:
        """
        Defined as the CoordinateSystem of the back face of the previous Surface composed with the coordinate break of
        this Surface.
        This is the CoordinateSystem used to calculate the thickness Vector and the location of the back face of the
        Surface.
        :return: Current CoordinateSystem of the Surface.
        """

        # If there is no previous Surface than the CoordinateSystem of this Surface is the same as the CoordinateSystem
        # of the global Component.
        # Otherwise the CoordinateSystem of this Surface is the composition of the previous Surface's CoordinateSystem
        # and the coordinate break of this Surface.
        if self.previous_surf is None:

            # If there is no global Component, return the global CoordinateSystem (no translation, no rotation)
            # Otherwise return the CoordinateSystem of the Component.
            if self.component is None:

                # Global CoordinateSystem
                return gcs()

            else:

                # Component CoordinateSystem
                return self.component.cs

        else:

            # Old current coordinate system composed with this Surface's coordinate break.
            return self.previous_surf.back_cs @ self.cs_break

    @property
    def front_cs(self) -> CoordinateSystem:

        return  self.previous_surf.back_cs @ (self.cs_break @ self.tilt_dec)


    @property
    def back_cs(self) -> CoordinateSystem:

        return self.cs + self.T

    def __str__(self) -> str:
        """
        :return: String representation of the surface
        """

        return self.name + ', comment = ' + self.comment + ', thickness = ' + str(self.thickness) + ', cs = [' \
               + self.cs.__str__() + ']'

