
from typing import Union, List
import numpy as np
import astropy.units as u
from beautifultable import BeautifulTable


import kgpy.optics
from kgpy.math import CoordinateSystem, Vector, quaternion
from kgpy.math.coordinate_system import GlobalCoordinateSystem as gcs
from kgpy.optics.surface.surface_type import SurfaceType, Standard
from kgpy.optics.surface import aperture

__all__ = ['Surface']


class Surface:
    """
    This class represents a single optical surface. This class should be a drop-in replacement for a Zemax surface, and
    have all the same properties and behaviors.
    """

    def __init__(self, name: str, thickness: u.Quantity = 0.0 * u.m):
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

        # Attributes to be set by the Component and System classes
        self.component = None                   # type: kgpy.optics.Component
        self.sys = None                         # type: kgpy.optics.System

        # Initialize links to neighboring surfaces
        self.prev_surf_in_system = None         # type: Surface
        self.next_surf_in_system = None         # type: Surface
        self.prev_surf_in_component = None      # type: Surface
        self.next_surf_in_component = None      # type: Surface

        # Save input arguments as class variables
        self.name = name
        self.thickness = thickness

        # Initialize other attributes
        self.comment = ''
        self.aperture = None        # type: aperture.Aperture

        # Coordinate breaks before/after surface
        self.before_surf_cs_break = gcs()
        self.after_surf_cs_break = gcs()

        # Space for storing previous evaluations of the coordinate systems.
        # We store this information instead of evaluating on the fly since the evaluations are expensive.
        # These variables must be reset to None if the system changes.
        # Todo: These variables should be moved to ZmxSurface, and the properties in this class need to be overwritten
        self._previous_cs = None
        self._cs = None
        self._front_cs = None
        self._back_cs = None

        # Additional ZOSAPI.Editors.LDE.ILDERow attributes to be set by the user
        # We need to modify private variables here so we don't inadvertently call properties in a subclass
        self.is_active = False
        self._is_stop = False
        self._radius = np.inf * u.mm    # type: u.Quantity
        self._surface_type = Standard           # type: SurfaceType

    @property
    def decenter_x(self) -> u.Quantity:

        x0 = self.before_surf_cs_break.X.x
        x1 = self.after_surf_cs_break.X.x

        if x0 != x1:
            raise ValueError('Decenter X undefined for different translations before and after surface')

        return x0

    @decenter_x.setter
    def decenter_x(self, val: u.Quantity) -> None:

        self.before_surf_cs_break.X.x = val
        self.after_surf_cs_break.X.x = -val

    @property
    def decenter_y(self) -> u.Quantity:

        y0 = self.before_surf_cs_break.X.y
        y1 = self.after_surf_cs_break.X.y

        if y0 != y1:
            raise ValueError('Decenter Y undefined for different translations before and after surface')

        return y0

    @decenter_y.setter
    def decenter_y(self, val: u.Quantity) -> None:

        self.before_surf_cs_break.X.y = val
        self.after_surf_cs_break.X.y = -val

    @property
    def decenter_z(self) -> u.Quantity:

        z0 = self.before_surf_cs_break.X.y
        z1 = self.after_surf_cs_break.X.y

        if z0 != z1:
            raise ValueError('Decenter Z undefined for different translations before and after surface')

        return z0

    @decenter_z.setter
    def decenter_z(self, val: u.Quantity) -> None:

        self.before_surf_cs_break.X.z = val
        self.after_surf_cs_break.X.z = -val

    @property
    def tilt_x(self) -> u.Quantity:

        x0 = self.before_surf_cs_break.R_x

        x1 = self.after_surf_cs_break.R_x

        if x0 != x1:
            raise ValueError('Tilt X undefined for different rotations before and after surface')

        return x0

    @tilt_x.setter
    def tilt_x(self, val: u.Quantity):

        self.before_surf_cs_break.R_x = val

        self.after_surf_cs_break.R_x = -val

    @property
    def tilt_y(self) -> u.Quantity:

        y0 = self.before_surf_cs_break.R_y

        y1 = self.after_surf_cs_break.R_y

        if y0 != y1:
            raise ValueError('Tilt Y undefined for different rotations before and after surface')

        return y0

    @tilt_y.setter
    def tilt_y(self, val: u.Quantity):

        self.before_surf_cs_break.R_y = val

        self.after_surf_cs_break.R_y = -val

    @property
    def tilt_z(self) -> u.Quantity:

        z0 = self.before_surf_cs_break.R_z

        z1 = self.after_surf_cs_break.R_z

        if z0 != z1:
            raise ValueError('Tilt Z undefined for different rotations before and after surface')

        return z0

    @tilt_z.setter
    def tilt_z(self, val: u.Quantity):

        self.before_surf_cs_break.R_z = val

        self.after_surf_cs_break.R_z = -val

    @property
    def surface_type(self):
        """
        Get the surface type: standard, paraxial, etc.

        :return: The type of this surface
        """
        return self._surface_type

    @surface_type.setter
    def surface_type(self, val: SurfaceType) -> None:
        """
        Set the surface type of the surface.

        :param val: New surface type.
        :return: None
        """
        self._surface_type = val

    @property
    def radius(self) -> u.Quantity:
        """
        Get the radius of curvature for this surface

        :return: The radius of curvature
        """
        return self._radius

    @radius.setter
    def radius(self, val: u.Quantity) -> None:
        """
        Set the radius of curvature of the front face of the surface

        :param val: New radius of curvature. Must have units of length.
        :return: None
        """
        if not isinstance(val, u.Quantity):
            raise ValueError('Radius is of type astropy.units.Quantity')

        if not val.unit.is_equivalent(u.m):
            raise ValueError('Radius must have dimensions of length')

        self._radius = val

    @property
    def thickness(self) -> u.Quantity:
        """
        :return: The distance between the front and back of this surface.
        """
        return self._thickness

    @thickness.setter
    def thickness(self, t: u.Quantity) -> None:
        """
        Set the thickness of the surface
        :param t: New surface thickness. Must have units of length
        :return: None
        """

        # Check that the thickness parameter has the units attribute
        if not isinstance(t, u.Quantity):
            raise TypeError('thickness parameter must be an astropy.units.Quantity')

        # Check that the thickness parameter has dimensions of length
        if not t.unit.is_equivalent(u.m):
            raise TypeError('thickness parameter does not have dimensions of length')

        # Update private storage variable
        self._thickness = t

        # Reset coordinate systems since they need to be reevaluated with the new thickness.
        self.reset_cs()

    @property
    def T(self) -> Vector:
        """
        Thickness vector
        :return: Vector pointing from the center of a surface's front face to the center of a surface's back face
        """
        return self.thickness * self.front_cs.zh

    @property
    def is_object(self) -> bool:
        """
        :return: True if this is the object surface in a system, False otherwise.
        If this surface is not associated with a system, this function returns False.
        """

        # If the surface is part of the system, check if it is the object surface
        if self.sys is not None:
            return self.sys.object == self

        # Otherwise, the surface is not part of a system, and we assume that it is not an object surface.
        else:
            return False

    @property
    def is_stop(self) -> bool:
        """
        Check if a surface is the stop surface in the optical system.

        :return: True if the surface is the stop, False otherwise.
        """
        return self._is_stop

    @is_stop.setter
    def is_stop(self, val: bool) -> None:
        """
        Set/unset this surface to be the stop surface.

        :param val: New value for the flag.
        :return: None
        """
        self._is_stop = val

    @property
    def is_image(self) -> bool:
        """
        :return: True if this is the image surface in a system, False otherwise.
        If this surface is not associated with a system, this function returns False.
        """

        # If the surface is part of the system, check if it is the image surface
        if self.sys is not None:
            return self.sys.image == self

        # Otherwise, the surface is not part of a system, and we assume that it is not an image surface.
        else:
            return False

    def _relative_list_element(self, surf_list: List['Surface'], rel_index: int) -> Union['Surface', None]:
        """
        Finds the surface from a relative index to this surface for a given list of surfaces.
        :param surf_list: The list of surfaces to index through
        :param rel_index: The index relative to this surface that we are interested in.
        :return: The surface at the given relative index if it exists, none otherwise.
        """

        # Find the index of this surface
        ind = self._find_self_in_list(surf_list)

        if ind is not None:

            # Compute the global index of the surface that we are interested int
            new_ind = ind + rel_index

            # If the global index is a valid index, return the surface at that index.
            if 0 <= new_ind < len(surf_list):
                return surf_list[new_ind]

        # Otherwise, the relative list element does not exist, and we return None.
        return None

    @property
    def system_index(self) -> int:
        """
        :return: The index of this surface within the overall optical system
        """

        return self._find_self_in_list(self.sys)

    @property
    def component_index(self) -> int:
        """
        :return: The index of this surface within it's component
        """

        return self._find_self_in_list(self.component)

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

    def reset_cs(self) -> None:
        """
        Resets the coordinate system for this surface and all surfaces after this surface in the system.

        This function is intended to be used by System.insert() to make sure that the coordinate systems get
        reinitialized appropriately after an new surface is inserted.
        :return: None
        """

        # Set the current surface to this surface
        surf = self

        # Loop until there are no more surfaces
        while True:

            # Reset all the coordinate systems
            surf._previous_cs = None
            surf._cs = None
            surf._front_cs = None
            surf._back_cs = None

            # If there is another surface in the system, update the current surface
            if surf.next_surf_in_system is not None:
                surf = surf.next_surf_in_system

            # Otherwise there are no surfaces left and we can break out of the loop.
            else:
                break

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

        # Only re-evaluate if the storage variable is unpopulated
        if self._previous_cs is None:

            # If this is not the first surface in the system
            if self.prev_surf_in_system is not None:

                # If the previous surface in the system is not the object system
                if not self.prev_surf_in_system.is_object:

                    # Grab a pointer to the coordinate system of the back of the previous surface
                    self._previous_cs = self.prev_surf_in_system.back_cs

                # If the previous surface in the system is the object surface.
                else:
                    self._previous_cs = gcs()

            # Otherwise this is the first surface in the system
            else:
                self._previous_cs = gcs()

        return self._previous_cs

    @property
    def cs(self) -> CoordinateSystem:
        """
        This coordinate system is the system associated with the face of a surface.
        For example: if the surface is a parabola, then the axis of rotation of the parabola would pass through the
        origin of this coordinate system.
        :return: Coordinate system of the surface face
        """

        # Only re-evaluate if the storage variable is unpopulated
        if self._cs is None:
            self._cs = self.previous_cs @ self.before_surf_cs_break

        return self._cs

    @property
    def front_cs(self) -> CoordinateSystem:
        """
        Coordinate system of the surface ignoring the local tilt/decenter of the surface.
        This is the coordinate system used to attach surfaces together.
        For example: if we needed to tilt a surface but keep the direction of propagation the same, this coordinate
        system is the original coordinate system of the surface.
        :return: Coordinate system of the surface face ignoring tilt/decenter.
        """

        # Only re-evaluate if the storage variable is unpopulated
        if self._front_cs is None:
            self._front_cs = self.cs @ self.after_surf_cs_break

        return self._front_cs

    @property
    def back_cs(self) -> CoordinateSystem:
        """
        Coordinate system of the back face of the surface.
        Found by translating the coordinate system of the front face (without the tilt/decenter) by the thickness
        vector.
        :return: Coordinate system of the back face of the surface.
        """

        # Only re-evaluate if the storage variable is unpopulated
        if self._back_cs is None:
            self._back_cs = self.front_cs + self.T

        return self._back_cs

    def __eq__(self, other: 'Surface'):
        """
        Check if two surface are equal by comparing all of their attributes
        :param other: The other surface to check against this one.
        :return: True if the two surfaces have the same values for all attributes, false otherwise.
        """
        a = self.name == other.name
        # b = self.comment == other.comment
        # d = self.thickness == other.thickness
        e = self.component == other.component

        return a and e

    @property
    def table_headers(self) -> List[str]:
        """
        List of headers used for printing table representation of surface

        :return: List of header strings
        """
        return ['Component', 'Surface', 'Thickness', '',
                'X_x', 'X_y', 'X_z', '',
                'T_x', 'T_y', 'T_z', '',
                'Is Stop']

    @property
    def table_row(self):
        """
        Format the surface as a list of strings for use in the table representation.

        :return: List of strings representing the surface.
        """

        # Only populate the component name string if the component exists
        if self.component is not None:
            comp_name = self.component.name
        else:
            comp_name = ''

        # Store shorthand versions of coordinate system variables
        cs = self.cs
        X = cs.X
        Q = cs.Q

        # Convert to Tait-Bryant angles
        T = quaternion.as_xyz_intrinsic_tait_bryan_angles(Q) * u.rad    # type: u.Quantity

        # Convert angles to degrees
        T = T.to(u.deg)

        def fmt(s) -> str:
            """
            Format a number for use with the table representation

            :param s: Number to be formatted
            :return: String representation of the number in the specified format.
            """
            return "{0:.2f}".format(s)

        return [comp_name, self.name, fmt(self.thickness), '',
                fmt(X.x), fmt(X.y), fmt(X.z), '',
                fmt(T[0]), fmt(T[1]), fmt(T[2]), '',
                self.is_stop]

    def __str__(self) -> str:
        """
        :return: String representation of the surface
        """

        # Construct new table for printing
        table = BeautifulTable(max_width=200)

        # Populate table
        table.column_headers = self.table_headers
        table.append_row(self.table_row)

        return table.get_string()

    __repr__ = __str__
