
import typing as tp
import numpy as np
import astropy.units as u
from beautifultable import BeautifulTable

from kgpy import optics, math

from . import Aperture, Material

__all__ = ['Surface']


class SurfaceCoordinateSystem(math.geometry.CoordinateSystem):

    def __init__(self, translation_first=True):

        super().__init__(translation_first)

        self.surface = None     # type: optics.system.configuration.Surface

class PreCoordinateSystem(SurfaceCoordinateSystem):

    @property
    def (self):
        return

    @.setter
    def (self, value):
        pass


class Surface:
    """
    This class represents a single optical surface. This class should be a drop-in replacement for a Zemax surface, and
    have all the same properties and behaviors.
    """

    def __init__(self, name: str):

        # Attributes to be set by the Component and System classes
        self.configuration = None

        # Save input arguments as class variables
        self.name = name
        self.thickness = 0.0 * u.m


        # Space for storing previous evaluations of the coordinate systems.
        # We store this information instead of evaluating on the fly since the evaluations are expensive.
        # These variables must be reset to None if the system changes.
        # Todo: These variables should be moved to ZmxSurface, and the properties in this class need to be overwritten
        self._pre_cs = None
        self._front_cs = None
        self._post_cs = None
        self._back_cs = None

        self.is_active = False
        self.is_object = False
        self.is_stop = False
        self.is_image = False

        self.radius = np.inf * u.mm
        self.conic = 0.0

        self.aperture = None
        self.material = None

    @property
    def configuration(self) -> tp.Optional[optics.system.Configuration]:
        return self._configuration

    @configuration.setter
    def configuration(self, value: tp.Optional[optics.system.Configuration]):
        self._configuration = value

    @property
    def material(self) -> Material:
        return self._material
    
    @material.setter
    def material(self, value: Material):
        self._material = value
        
    @property
    def conic(self) -> u.Quantity:
        return self._conic
    
    @conic.setter
    def conic(self, value: u.Quantity):
        self._conic = value

    @property
    def aperture(self) -> Aperture:
        return self._aperture

    @aperture.setter
    def aperture(self, value: Aperture):
        self._aperture = value

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
        self._radius = val

    @property
    def thickness(self) -> u.Quantity:
        """
        :return: The distance between the front and back of this surface.
        """
        return self._thickness

    @thickness.setter
    def thickness(self, value: u.Quantity) -> None:
        """
        Set the thickness of the surface
        :param value: New surface thickness. Must have units of length
        :return: None
        """

        # Update private storage variable
        self._thickness = value

        # Reset coordinate systems since they need to be reevaluated with the new thickness.
        self.update()

    @property
    def T(self) -> math.geometry.Vector:
        """
        Thickness vector
        :return: Vector pointing from the center of a surface's front face to the center of a surface's back face
        """
        return self.thickness * self.post_cs.z_hat

    @property
    def is_object(self) -> bool:
        return self._is_object

    @is_object.setter
    def is_object(self, value: bool):
        self._is_object = value

    @property
    def is_stop(self) -> bool:
        return self._is_stop

    @is_stop.setter
    def is_stop(self, value: bool):
        self._is_stop = value

    @property
    def is_image(self) -> bool:
        return self._is_image

    @is_image.setter
    def is_image(self, value: bool):
        self._is_image = value

    @property
    def pre_cs(self) -> math.geometry.CoordinateSystem:
        return self._pre_cs

    @pre_cs.setter
    def pre_cs(self, value: math.geometry.CoordinateSystem):
        self._pre_cs = value

        self.front_cs = self.front_cs_

    @property
    def pre_cs_(self):

        try:
            return self.previous_surface.back_cs

        except AttributeError:
            return math.geometry.CoordinateSystem()

    @property
    def front_cs(self) -> math.geometry.CoordinateSystem:
        return self._front_cs

    @front_cs.setter
    def front_cs(self, value: math.geometry.CoordinateSystem):
        self._front_cs = value

        self.post_cs = self.post_cs_

    @property
    def front_cs_(self) -> math.geometry.CoordinateSystem:
        return self.pre_cs @ self.pre_tilt_decenter

    @property
    def back_cs(self) -> math.geometry.CoordinateSystem:
        return self._back_cs

    @back_cs.setter
    def back_cs(self, value: math.geometry.CoordinateSystem):
        self._back_cs = value

        try:
            self.next_surface.pre_cs = self.next_surface.pre_cs_

        except AttributeError:
            pass

    @property
    def back_cs_(self):
        return self.post_cs + self.T



    @property
    def pre_cs(self) -> math.geometry.CoordinateSystem:
        self._front_cs = self._pre_cs @ self.pre_tilt_decenter

        return self._pre_cs


    @property
    def cs(self) -> math.geometry.CoordinateSystem:

        # Only re-evaluate if the storage variable is unpopulated
        if self._front_cs is None:
            pre_tilt_decenter

        return self._front_cs

    @property
    def post_cs(self) -> math.geometry.CoordinateSystem:

        # Only re-evaluate if the storage variable is unpopulated
        if self._front_cs is None:
            self._front_cs = self.cs @ self.post_tilt_decenter

        return self._front_cs

    @property
    def back_cs(self) -> math.geometry.CoordinateSystem:

        # Only re-evaluate if the storage variable is unpopulated
        if self._back_cs is None:
            self._back_cs = self.post_cs + self.T

        return self._back_cs



    @property
    def table_headers(self) -> tp.List[str]:
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
        X = cs.translation
        Q = cs.rotation

        # Convert to Tait-Bryant angles
        T = kgpy.math.geometry.quaternion.quaternion.as_xyz_intrinsic_tait_bryan_angles(Q) * u.rad    # type: u.Quantity

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

    def update(self) -> None:
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
            surf._pre_cs = None
            surf._front_cs = None
            surf._front_cs = None
            surf._back_cs = None

            # If there is another surface in the system, update the current surface
            if surf.next_surf_in_system is not None:
                surf = surf.next_surf_in_system

            # Otherwise there are no surfaces left and we can break out of the loop.
            else:
                break

    @property
    def index(self) -> int:
        """
        :return: The index of this surface within the overall optical system
        """

        return self.configuration.index(self)

    @property
    def previous_surface(self) -> tp.Optional['Surface']:

        i = self.index

        if i != 0:
            return self.configuration[i - 1]

        return None

    @property
    def next_surface(self):

        i = self.index
        n_surf = len(self.configuration)

        if (i + 1) % n_surf != 0:

            return self.configuration[i + 1]

        return None

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
