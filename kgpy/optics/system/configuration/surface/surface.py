
import typing as tp
import numpy as np
import astropy.units as u
from beautifultable import BeautifulTable

from kgpy import optics, math

from . import Aperture, Material

__all__ = ['Surface']


class Surface:
    """
    This class represents a single optical surface. This class should be a drop-in replacement for a Zemax surface, and
    have all the same properties and behaviors.
    """

    def __init__(self,
                 name: str = '',
                 is_stop: bool = False,
                 thickness: u.Quantity = None,
                 ):

        if thickness is None:
            thickness = 0 * u.m

        self._name = name
        self._is_stop = is_stop
        self._thickness = thickness

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_stop(self) -> bool:
        return self._is_stop

    @property
    def thickness(self) -> u.Quantity:
        return self._thickness


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
        T = math.geometry.quaternion.quaternion.as_xyz_intrinsic_tait_bryan_angles(Q) * u.rad    # type: u.Quantity

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
