
from unittest import TestCase
from numbers import Real
from typing import Union, List, Tuple
import numpy as np
import quaternion as q
import astropy.units as u
from astropy.coordinates import Distance


import kgpy.optics
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
        # These are links to the previous/next surface in the component, previous/next surface in the system, and a link
        # to the root component.
        # These are used to recursively calculate properties of this surface instead of explicitly updating
        # this surface.
        # For example if the thickness of an earlier surface changes, we would have to remember to update this surface
        # with a new global position.
        # However if we calculate the global position by adding up the thickness of all previous surfaces, it is not
        # necessary to remember to update anything.
        self.prev_surf_in_system = None        # type: Surface
        self.next_surf_in_system = None        # type: Surface
        self.prev_surf_in_component = None     # type: Surface
        self.next_surf_in_component = None     # type: Surface
        self.component = None                  # type: kgpy.optics.Component

    @property
    def system_index(self):
        """
        :return: The index of this surface within the overall optical system
        """

        # If there is already a previous surface in the system, the index of this surface is the index of the previous
        # surface incremented by one.
        # Otherwise, there is not a previous surface in the system and this is the first surface, so the index is zero.
        if self.prev_surf_in_system is not None:
            return self.prev_surf_in_system.system_index + 1
        else:
            return 0

    @property
    def component_index(self):
        """
        :return: The index of this surface within it's component
        """

        # If there is not another surface in this component, the index is zero, otherwise the component index is the
        # index of the previous surface in the component incremented by one
        if self.prev_surf_in_component is None:
            return 0
        else:
            return self.prev_surf_in_component.component_index + 1

    @property
    def T(self) -> Vector:
        """
        Thickness vector
        :return: Vector pointing from the center of a surface's front face to the center of a surface's back face
        """
        return self.thickness * self.front_cs.zh

    @property
    def previous_cs(self) -> CoordinateSystem:
        """
        The coordinate system of the surface before this surface in the optical system.
        This is the coordinate system that this surface will be "attached" to.
        There is a hierarchy to this function as follows: last surface in system -> last surface in component -> first
        surface in component -> first surface in system.
        So the coordinate system previous to this surface is determined by the hierarchy above.
        :return: The last CoordinateSystem in the optical system
        """

        if self.prev_surf_in_system is not None:

            cs = self.prev_surf_in_system.back_cs

            if self.component is not None:

                if self.prev_surf_in_component is not None:

                    return cs

                else:

                    return cs @ self.component.cs_break

            else:

                return cs

        else:

            if self.component is not None:

                if self.prev_surf_in_component is not None:

                    return self.prev_surf_in_component.back_cs

                else:

                    return self.component.cs_break

            else:

                return gcs()

    @property
    def cs(self) -> CoordinateSystem:
        """
        This coordinate system is the system associated with the face of a surface.
        For example: if the surface is a parabola, then the axis of rotation of the parabola would pass through the
        origin of this coordinate system.
        :return: Coordinate system of the surface face
        """

        return self.previous_cs @ (self.cs_break @ self.tilt_dec)

    @property
    def front_cs(self) -> CoordinateSystem:
        """
        Coordinate system of the surface ignoring the local tilt/decenter of the surface.
        This is the coordinate system used to attach surfaces together.
        For example: if we needed to tilt a surface but keep the direction of propagation the same, this coordinate
        system is the original coordinate system of the surface.
        :return: Coordinate system of the surface face ignoring tilt/decenter.
        """

        return self.previous_cs @ self.cs_break


    @property
    def back_cs(self) -> CoordinateSystem:
        """
        Coordinate system of the back face of the surface.
        Found by translating the coordinate system of the front face (without the tilt/decenter) by the thickness
        vector.
        :return: Coordinate system of the back face of the surface.
        """

        return self.front_cs + self.T

    def __str__(self) -> str:
        """
        :return: String representation of the surface
        """

        return 'surface(' + self.name + ', comment = ' + self.comment + ', thickness = ' + str(self.thickness) \
               + ', ' + self.cs.__str__() + ')'


    def __repr__(self):

        return self.__str__()