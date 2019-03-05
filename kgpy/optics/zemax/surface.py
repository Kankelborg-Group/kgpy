
from win32com.client import CastTo
from typing import List, Dict, Union, Type, Any
import astropy.units as u
from collections import OrderedDict
from enum import IntEnum, auto

import kgpy.optics
from kgpy.math.quaternion import *
from kgpy.math import Vector, CoordinateSystem
from kgpy.math.coordinate_system import GlobalCoordinateSystem as gcs
from kgpy.optics import Surface, Component, System
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.ZOSAPI.Editors.LDE import ILDERow

__all__ = ['ZmxSurface']


class ZmxSurface(Surface):
    """
    Child of the Surface class which acts as a wrapper around the Zemax API ILDERow class (a Zemax surface).
    The ZmxSurface class is intended to be used temporarily as the functionality in the ILDERow class is implemented in
    the Surface superclass.
    """

    main_str = 'main'
    cs_break_str = 'cs_break'
    tilt_dec_str = 'tilt_dec'
    tilt_dec_return_str = 'tilt_dec_return'
    thickness_str = 'thickness'

    class AttrPriority(IntEnum):
        """
        Enumeration class to describe the priority that each attribute gets when the surface is written to file.
        """
        cs_break = auto()
        tilt_dec = auto()
        main = auto()
        tilt_dec_return = auto()
        thickness = auto()

    # Dictionary that maps the attribute strings to their priority level
    priority_dict = {
        main_str:               AttrPriority.main,
        cs_break_str:           AttrPriority.cs_break,
        tilt_dec_str:           AttrPriority.tilt_dec,
        tilt_dec_return_str:    AttrPriority.tilt_dec_return,
        thickness_str:          AttrPriority.thickness
    }

    # noinspection PyMissingConstructor
    def __init__(self, name: str, attr_rows: 'OrderedDict[str, ILDERow]', length_units: u.Unit):
        """
        Constructor for ZmxSurface object.
        :param attr_rows: A Dictionary of attributes where the key is the attribute name and the value is the ILDERow
        associated with the attribute.
        Note that this structure should contain a main attribute/row, which is the default a
        :param length_units: The units that this row uses for distance measurements.
        """

        # Check that the attribute rows has a main field
        if self.main_str not in attr_rows:
            raise ValueError('attr_rows must contain a main field')

        # Initialize attribute dictionary
        self._attr_rows_var = OrderedDict()  # type: OrderedDict[str, ILDERow]

        # Save arguments to class variables
        self.name = name
        self._attr_rows = attr_rows
        self.u = length_units

        # Initialize other attributes
        self.comment = ''

        # Attributes to be set by Component and System classes
        self.component = None                   # type: Component
        self.sys = None                         # type: kgpy.optics.ZmxSystem

    @property
    def thickness(self) -> u.Quantity:
        """
        :return: Thickness of the surface in lens units
        """

        # If there is a thickness row defined, return the value from that row
        if self.thickness_str in self._attr_rows:
            return self._attr_rows[self.thickness_str].Thickness * self.u

        # Otherwise, return the thickness value from the main row.
        else:
            return self._attr_rows[self.main_str].Thickness * self.u

    @thickness.setter
    def thickness(self, t: u.Quantity) -> None:
        """
        Set the thickness of the surface
        :param t: New thickness of surface
        :return: None
        """

        # If there is a thickness row defined, update the value from that row
        if self.thickness_str in self._attr_rows:
            self._attr_rows[self.thickness_str].Thickness = float(t / self.u)

        # Otherwise, update the thickness value from the main row.
        else:
            self._attr_rows[self.main_str].Thickness = float(t / self.u)

    @property
    def cs_break(self) -> CoordinateSystem:
        """
        Compute the coordinate system for this coordinate break.
        Zemax represents the tilts of a optical surface using the xyz intrinsic Tait-Bryan angles.
        This function tries to find a coordinate break surface within the list of attribute rows, and if it exists,
        constructs a new coordinate system to represent the coordinate break.
        If the coordinate break does not exist, it returns the identity coordinate system.
        :return: A coordinate system representing a Zemax coordinate break.
        """

        # If a cs_break row is defined, convert to a CoordinateSystem object and return.
        if self.cs_break_str in self._attr_rows:

            # Find the Zemax LDE row corresponding to the coordinate break for this surface
            row = self._attr_rows[self.cs_break_str]

            # Extract the tip/tilt/decenter data from the Zemax row
            data = self._ISurfaceCoordinateBreak_data(row)

            # Construct vector from decenter parameters
            X = Vector([data.Decenter_X, data.Decenter_Y, 0] * self.u)

            # Construct rotation quaternion from tilt parameters
            Q = from_xyz_intrinsic_tait_bryan_angles(data.TiltAbout_X, data.TiltAbout_Y, data.TiltAbout_Z)

            # Determine which order the tilts/decenters should be executed
            if data.Order == 0:
                translation_first = True
            else:
                translation_first = False

            # Create new coordinate system representing the Zemax coordinate break
            cs = CoordinateSystem(X, Q, translation_first=translation_first)

            return cs

        # Otherwise just return a identity coordinate system
        else:
            return gcs()

    @cs_break.setter
    def cs_break(self, cs: CoordinateSystem) -> None:
        """
        Set a Zemax coordinate break using a CoordinateSystem
        :param cs: CoordinateSystem to convert to a Zemax coordinate break
        :return: None
        """

        # If there is not already a coordinate break row defined, create it
        if self.cs_break_str not in self._attr_rows:
            self._insert_attr_row(self.cs_break_str, row_type=ZOSAPI.Editors.LDE.ISurfaceCoordinateBreak)

        # Find the Zemax LDE row corresponding to the coordinate break for this surface
        row = self._attr_rows[self.cs_break_str]

        # Extract the tip/tilt/decenter data from the Zemax row
        data = self._ISurfaceCoordinateBreak_data(row)

        # Check that we have a coordinate system that is compatible with a coordinate break
        if cs.X.z != 0:
            raise ValueError('Nonzero z-translation')

        # Update the order
        if cs.translation_first:
            data.Order = 0
        else:
            data.Order = 1

        # Update the translation
        data.Decenter_X = cs.X.x / self.u
        data.Decenter_Y = cs.X.y / self.u

        # Update the rotation
        a, b, c = as_xyz_intrinsic_tait_bryan_angles(cs.Q)
        data.TiltAbout_X = a
        data.TiltAbout_Y = b
        data.TiltAbout_Z = c

    @staticmethod
    def _ISurfaceCoordinateBreak_data(row: ILDERow) -> ZOSAPI.Editors.LDE.ISurfaceCoordinateBreak:
        """
        Find the Coordinate Break SurfaceData of a provided row.
        If the provided row is not a Coordinate Break, a ValueError will be thrown
        :param row: Row to find the CoordinateBreak SurfaceData for.
        :return: CoordinateBreak SurfaceData for the provided row
        """

        # If the row is a CoordinateBreak surface, update the parameters with the new value from the provided
        # coordinate system.
        if row.Type == ZOSAPI.Editors.LDE.SurfaceType.CoordinateBreak:

            # Return the coordinate break parameters
            return CastTo(row.SurfaceData, ZOSAPI.Editors.LDE.ISurfaceCoordinateBreak.__name__
                          )  # type: ZOSAPI.Editors.LDE.ISurfaceCoordinateBreak

        # Otherwise, the row is not a coordinate break surface, and this is unexpected
        else:
            raise ValueError('Row is not of type', ZOSAPI.Editors.LDE.ISurfaceCoordinateBreak.__name__)

    @property
    def tilt_dec(self) -> CoordinateSystem:

        # Find the LDE row corresponding to tilt/decenter
        row = self._get_attr_row('tilt_dec')

        if row is not self._row:

            # Find the LDE row corresponding to tilt/decenter return
            ret = self._get_attr_row('tilt_dec_return')

    @property
    def _attr_rows(self) -> 'OrderedDict[str, ILDERow]':
        """
        :return: An ordered dict of attributes, where the attribute name is the key and the Zemax row is the value.
        """
        return self._attr_rows_var

    @_attr_rows.setter
    def _attr_rows(self, attr_rows: 'OrderedDict[str, ILDERow]') -> None:
        """
        Write to the ordered dict of attributes, checking that all the Zemax rows are consecutive and that the surfaces
        are ordered in the correct priority
        :param attr_rows: Ordered dict of attributes, where the key is the attribute name and the value is a Zemax row.
        :return: None
        """

        # Initial vales for the last index and last priority
        last_ind = -1
        last_pri = -1

        # Loop through all the attributes in the dictionary and check that each element is consecutive and in ascending
        # priority.
        for attr, row in self._attr_rows.items():

            # Extract the index and priority values for comparison
            ind = row.RowIndex
            pri = self.priority_dict[attr]

            # Check that this index is one greater than the previous index
            if ind != last_ind + 1:
                raise ValueError('Non-consecutive attribute rows')

            # Check that this priority is greater than or equal to the previous priority
            if pri < last_pri:
                raise ValueError('Higher priority surface before lower priority surface')

        # If there were no errors raised in the previous loop, we are free to write to the variable
        self._attr_rows_var = attr_rows

    def _insert_attr_row(self, attr_name: str, row_type: Type[Any] = ZOSAPI.Editors.LDE.ISurfaceStandard) -> None:
        """
        Insert a new attribute row, with it's position specified by the priority of the attribute
        :param attr_name: Name of the attribute, determines priority
        :param row_type: The type of ILDErow that should be associated with the attribute
        :return:
        """

        # Check that the attribute doesn't already have an associated Zemax row
        if attr_name in self._attr_rows:
            raise ValueError('Attribute already defined')

        # Extract the priority of the provided attribute
        pri = self.priority_dict[attr_name]

        # Create a new ordered dict to store the result.
        # Since OrderedDict does not have an insert method, we create a new instance and insert the new surface as
        # we're copying all the keys/values
        attr_rows = OrderedDict()

        # Loop through all the attributes, find where the new attribute should go, and insert the new attribute
        for attr, row in self._attr_rows.items():

            # If the priority of the provided attribute is less than the current row, this is the correct index for
            # the new attribute row.
            if pri < self.priority_dict[attr]:

                # Insert a new row at the index of the current row
                new_row = self.sys.zmx_sys.LDE.InsertNewSurfaceAt(row.RowIndex)

                # Change the type of the row to the requested type
                new_row.ChangeType(new_row.GetSurfaceTypeSettings(row_type))

                # Add the new row to the attribute dict
                attr_rows[attr_name] = new_row

            # Populate the new dict with key/value pairs from the old dict
            attr_rows[attr] = row

        # Update the attribute row dictionary
        self._attr_rows = attr_rows

    @property
    def _attr_rows_indices(self) -> List[int]:
        """
        Todo: There is probably a list comprehension way to handle this
        :return: The index of each ILDERow in the attribute dictionary
        """

        # Space for storing the result
        indices = []

        # Loop over all items in the dictionary and append row index to output
        for _, row in self._attr_rows.items():
            indices.append(row.RowIndex)

        return indices

    @property
    def first_row_ind(self):
        """
        :return: The index of the first row of this surface in the Zemax file.
        """
        return self._attr_rows_indices[0]

    @property
    def last_row_ind(self):
        """
        :return: The index of the last row of this surface in the Zemax file.
        """
        return self._attr_rows_indices[-1]








