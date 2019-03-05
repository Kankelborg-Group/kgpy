
from typing import Union, List, Tuple
import numpy as np
import quaternion as q
import astropy.units as u


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

        # Initialize private variables
        self._thickness = 0 * u.mm

        # Save input arguments as class variables
        self.name = name
        self.thickness = thickness
        self.comment = comment
        self.tilt_dec = tilt_dec
        self.cs_break = cs_break

        # Attributes to be set by the Component and System classes
        self.component = None                   # type: kgpy.optics.Component
        self.sys = None                         # type: kgpy.optics.System

        # Additional ZOSAPI.Editors.LDE.ILDERow attributes to be set by the user
        self.is_active = False
        self.is_image = False
        self.is_stop = False
        self.is_object = False

    @property
    def prev_surf_in_system(self) -> Union['Surface', None]:
        """
        :return: The surface before this surface in the system
        """

        # If the system is defined, find the previous surface
        if self.sys is not None:
            return self._relative_list_element(self.sys.surfaces, -1)

        # Otherwise there is no system defined and we return none
        else:
            return None

    @property
    def next_surf_in_system(self) -> Union['Surface', None]:
        """
        :return: The surface after this surface in the system
        """

        # If the system is defined, find the next surface
        if self.sys is not None:
            return self._relative_list_element(self.sys.surfaces, 1)

        # Otherwise there is no system defined and we return none
        else:
            return None

    @property
    def prev_surf_in_component(self) -> Union['Surface', None]:
        """
        :return: The surface before this surface in the component.
        """

        # If the component is defined, find the previous surface
        if self.component is not None:
            return self._relative_list_element(self.component.surfaces, -1)

        # Otherwise there is no component defined and we return none
        else:
            return None

    @property
    def next_surf_in_component(self) -> Union['Surface', None]:
        """
        :return: The surface after this surface in the component
        """

        # If the component is defined, find the next surface
        if self.component is not None:
            return self._relative_list_element(self.component.surfaces, 1)

        # Otherwise there is no component defined and we return none
        else:
            return None

    def _relative_list_element(self, surf_list: List['Surface'], rel_index: int) -> Union['Surface', None]:
        """
        Finds the surface from a relative index to this surface for a given list of surfaces.
        :param surf_list: The list of surfaces to index through
        :param rel_index: The index relative to this surface that we are interested in.
        :return: The surface at the given relative index if it exists, none otherwise.
        """

        # Find the index of this surface
        ind = self._find_self_in_list(surf_list)

        # Compute the global index of the surface that we are interested int
        new_ind = ind + rel_index

        # If the global index is a valid index, return the surface at that index.
        if 0 <= new_ind < len(surf_list):
            return surf_list[new_ind]

        # Otherwise, the global index is not valid, and we return None.
        else:
            return None

    @property
    def system_index(self) -> int:
        """
        :return: The index of this surface within the overall optical system
        """

        return self._find_self_in_list(self.sys.surfaces)

    @property
    def component_index(self) -> int:
        """
        :return: The index of this surface within it's component
        """

        return self._find_self_in_list(self.component.surfaces)

    def _find_self_in_list(self, surf_list: List['Surface']) -> Union[int, None]:
        """
        Find the index of this surface in the provided list
        :param surf_list: List to be searched for this surface
        :return: The index of the surface if it exists in the list, None otherwise.
        """

        # Make an empty list to store the indices matching this surface
        ind = []

        # Loop through the provided list to check for any matches
        for s, surf in enumerate(surf_list):

            # If there are any matches append the index to our list of indices
            if self == surf:
                ind.append(s)

        # If there were no matching surfaces found, return None
        if len(ind) == 0:
            return None

        # If there was one matching surface found, return it's index
        elif len(ind) == 1:
            return ind[0]

        # Otherwise, there was more than one matching surface found, and this is unexpected
        else:
            raise ValueError('More than one surface matches in list')

    @property
    def thickness(self) -> u.Quantity:
        """
        :return: The distance between the front and back of this surface.
        """
        return self._thickness

    @thickness.setter
    def thickness(self, t: u.Quantity) -> None:
        """
        Set the
        :param t:
        :return:
        """

        # Check that the thickness parameter has the units attribute
        if not isinstance(t, u.Quantity):
            raise TypeError('thickness parameter must be an astropy.units.Quantity')

        # Check that the thickness parameter has dimensions of length
        if not t.unit.is_equivalent(u.m):
            raise TypeError('thickness parameter does not have dimensions of length')

        self._thickness = t

    @property
    def T(self) -> Vector:
        """
        Thickness vector
        :return: Vector pointing from the center of a surface's front face to the center of a surface's back face
        """
        return self.thickness * self.front_cs.zh

    @property
    def _previous_cs(self) -> CoordinateSystem:
        """
        The coordinate system of the surface before this surface in the optical system.
        This is the coordinate system that this surface will be "attached" to.
        There is a hierarchy to this function as follows: last surface in system -> last surface in component -> first
        surface in component -> first surface in system.
        So the coordinate system previous to this surface is determined by the hierarchy above.
        :return: The last CoordinateSystem in the optical system
        """

        # If this is not the first surface in the system
        if self.prev_surf_in_system is not None:

            # Grab a pointer to the coordinate system of the back of the previous surface
            cs = self.prev_surf_in_system.back_cs

            # If this surface belongs to a component
            if self.component is not None:

                # If this is not the first surface in the component
                if self.prev_surf_in_component is not None:

                    # Return the coordinate system of the back of the previous surface
                    return cs

                # Otherwise this is the first surface in the component
                else:

                    # And we return the coordinate system of the back of the previous surface, composed with the
                    # coordinate break of this component
                    return cs @ self.component.cs_break

            # Otherwise, this surface is not associated with a component, and we can just return the coordinate system
            # of the back of the previous surface
            else:
                return cs

        # Otherwise this is the first surface in the system
        else:

            # If this surface belongs to a component
            if self.component is not None:

                # If this is not the first surface in the component
                if self.prev_surf_in_component is not None:

                    # Return the coordinate system of the back of the previous surface in the component
                    return self.prev_surf_in_component.back_cs

                # Otherwise, this is the first surface in the component
                else:

                    # And we return the coordinate break of this component
                    return self.component.cs_break

            # Otherwise this surface does not belong to a component
            else:

                # And it lives at the origin
                return gcs()

    @property
    def cs(self) -> CoordinateSystem:
        """
        This coordinate system is the system associated with the face of a surface.
        For example: if the surface is a parabola, then the axis of rotation of the parabola would pass through the
        origin of this coordinate system.
        :return: Coordinate system of the surface face
        """

        return self._previous_cs @ (self.cs_break @ self.tilt_dec)

    @property
    def front_cs(self) -> CoordinateSystem:
        """
        Coordinate system of the surface ignoring the local tilt/decenter of the surface.
        This is the coordinate system used to attach surfaces together.
        For example: if we needed to tilt a surface but keep the direction of propagation the same, this coordinate
        system is the original coordinate system of the surface.
        :return: Coordinate system of the surface face ignoring tilt/decenter.
        """

        return self._previous_cs @ self.cs_break

    @property
    def back_cs(self) -> CoordinateSystem:
        """
        Coordinate system of the back face of the surface.
        Found by translating the coordinate system of the front face (without the tilt/decenter) by the thickness
        vector.
        :return: Coordinate system of the back face of the surface.
        """

        return self.front_cs + self.T

    def __eq__(self, other: 'Surface'):
        """
        Check if two surface are equal by comparing all of their attributes
        :param other: The other surface to check against this one.
        :return: True if the two surfaces have the same values for all attributes, false otherwise.
        """
        a = self.name == other.name
        b = self.comment == other.comment
        d = self.thickness == other.thickness
        e = self.component == other.component

        return a and b and d and e

    def __str__(self) -> str:
        """
        :return: String representation of the surface
        """

        # Only populate the component name string if the component exists
        if self.component is not None:
            comp_name = self.component.name
        else:
            comp_name = ''

        # Construct the return string
        return 'surface(' + comp_name + '.' + self.name + ', thickness = ' + str(self.thickness) \
               + ', ' + self._previous_cs.__str__() + ')'

    __repr__ = __str__
