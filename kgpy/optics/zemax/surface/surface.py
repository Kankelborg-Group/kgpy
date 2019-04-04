
from win32com.client import CastTo
from typing import List, Dict, Union, Type, Any, Tuple, Iterator
import astropy.units as u
from collections import OrderedDict
from enum import IntEnum, auto

import kgpy.optics
from kgpy.math.quaternion import *
from kgpy.math import Vector, CoordinateSystem
from kgpy.math.coordinate_system import GlobalCoordinateSystem as gcs
from kgpy.optics import Surface, surface
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.surface.aperture import Circular, Rectangular, Spider
from kgpy.optics.zemax.ZOSAPI.Editors.LDE import ILDERow

__all__ = ['ZmxSurface']


class ZmxSurface(Surface):
    """
    Child of the Surface class which acts as a wrapper around the Zemax API ILDERow class (a Zemax surface).
    The ZmxSurface class is intended to be used temporarily as the functionality in the ILDERow class is implemented in
    the Surface superclass.
    """

    main_str = 'main'
    before_surf_cs_break_str = 'before_surf_cs_break'
    after_surf_cs_break_str = 'after_surf_cs_break'
    thickness_str = 'thickness'
    stop_str = 'stop'
    aper_str = 'aper'
    mech_aper_str = 'mech_aper'

    class AttrPriority(IntEnum):
        """
        Enumeration class to describe the priority that each attribute gets when the surface is written to file.
        """

        before_surf_cs_break = auto()
        main = auto()
        aper = main
        mech_aper = main
        after_surf_cs_break = auto()
        thickness = auto()

    # Dictionary that maps the attribute strings to their priority level
    priority_dict = {
        main_str:                   AttrPriority.main,
        before_surf_cs_break_str:   AttrPriority.before_surf_cs_break,
        after_surf_cs_break_str:    AttrPriority.after_surf_cs_break,
        thickness_str:              AttrPriority.thickness,
        aper_str:                   AttrPriority.aper,
        mech_aper_str:              AttrPriority.mech_aper
    }

    def __init__(self, name: str, thickness: u.Quantity = 0.0 * u.m):
        """
        Constructor for the ZmxSurface class.
        This constructor places the surface at the origin of the global coordinate system, it needs to be moved into
        place after the call to this function.
        :param name: Human-readable name of the surface
        :param thickness: Thickness of the surface along the self.cs.zh direction. Must have dimensions of length.
        """

        # Initialize list of ILDERows associated with class attributes.
        # This needs to be before the superclass constructor so properties such as thickness are defined
        self._attr_rows = {}

        # Call superclass constructor
        super().__init__(name, thickness)

        # Override the type of the system pointer
        self.sys = None     # type: kgpy.optics.ZmxSystem

        # Override the type of the neighboring surfaces
        self.prev_surf_in_system = None         # type: ZmxSurface
        self.next_surf_in_system = None         # type: ZmxSurface
        self.prev_surf_in_component = None      # type: ZmxSurface
        self.next_surf_in_component = None      # type: ZmxSurface

    @staticmethod
    def from_surface(surf: Surface) -> 'ZmxSurface':
        """
        Convert a Surface object to ZmxSurface object
        :param surf: Surface object to convert
        :return: A new ZmxSurface object equal to the provided Surface object
        """

        # Construct new ZmxSurface
        zmx_surf = ZmxSurface(surf.name)

        # Copy remaining attributes
        zmx_surf.comment = surf.comment
        zmx_surf.thickness = surf.thickness
        zmx_surf.before_surf_cs_break = surf.before_surf_cs_break
        zmx_surf.after_surf_cs_break = surf.after_surf_cs_break
        zmx_surf.component = surf.component
        zmx_surf.aperture = surf.aperture

        return zmx_surf

    @staticmethod
    def from_attr_dict(surf_name: str, attr_dict: Dict[str, ILDERow]) -> 'ZmxSurface':

        # Construct new ZmxSurface
        zmx_surf = ZmxSurface(surf_name)

        # Set the attribute surfaces with the provided value
        zmx_surf._attr_rows = attr_dict

        # Initialize private variables associated with each property
        zmx_surf._thickness = None
        zmx_surf._tilt_dec = None
        zmx_surf._before_surf_cs_break = None
        zmx_surf._after_surf_cs_break = None
        zmx_surf._is_stop = None
        zmx_surf._radius = None

        return zmx_surf

    @property
    def aperture(self) -> surface.Aperture:
        return self._aperture

    @aperture.setter
    def aperture(self, aper: surface.Aperture):

        if self.sys is not None:

            if isinstance(aper, surface.aperture.Circular):

                self._aperture = Circular(aper.min_radius, aper.max_radius, self)

            elif isinstance(aper, surface.aperture.Rectangular):

                self._aperture = Rectangular(aper.half_width_x, aper.half_width_y, self)

            elif isinstance(aper, surface.aperture.Spider):

                self._aperture = Spider(aper.arm_width, aper.num_arms, self)

        else:

            self._aperture = aper

    @property
    def main_row(self):
        return self._attr_rows[self.main_str]

    @property
    def radius(self) -> u.Quantity:
        """
        Get radius of curvature this optic, in the units of the Zemax system

        :return: Radius of curvature
        """

        # If value unpopulated, read from Zemax
        if self._radius is None:
            self._radius = self._attr_rows[self.main_str].Radius * self.sys.lens_units

        return self._radius

    @radius.setter
    def radius(self, val: u.Quantity) -> None:
        """
        Set the radius of curvature for this surface.

        :param val: New radius of curvature, must have units of length
        :return: None
        """

        if not isinstance(val, u.Quantity):
            raise ValueError('Radius is of type astropy.units.Quantity')

        if not val.unit.is_equivalent(u.m):
            raise ValueError('Radius must have dimensions of length')

        self._radius = val
        self._attr_rows[self.main_str].Radius = float(val / self.sys.lens_units)

    @property
    def is_stop(self) -> bool:
        """
        :return: True if this surface contains a stop row, False otherwise
        """

        # If the surface is part of a ZOS system, return the stop flag of the main surface
        if self.sys is not None:

            # If the stop state has not been populated, read from zemax
            if self._is_stop is None:
                self._is_stop = self._attr_rows[self.main_str].IsStop

            return self._is_stop

        # Otherwise this surface is not part of a ZOS system and is not the stop
        else:
            return False

    @is_stop.setter
    def is_stop(self, val: bool):
        """
        Set the stop surface to the provided value
        :param val: True if this surface is the stop surface, False otherwise
        :return: None
        """

        # The surface can only be the stop if
        self._attr_rows[self.main_str].IsStop = val
        self._is_stop = val

    @property
    def thickness(self) -> u.Quantity:
        """
        :return: Thickness of the surface in lens units
        """

        # If value has not been populated, read from Zemax
        if self._thickness is None:

            # If there is a thickness row defined, return the value from that row
            if self.thickness_str in self._attr_rows:
                self._thickness = self._attr_rows[self.thickness_str].Thickness * self.sys.lens_units

            # Otherwise, return the thickness value from the last row in the surface
            else:
                self._thickness = self._attr_rows_list[-1][1].Thickness * self.sys.lens_units

        return self._thickness

    @thickness.setter
    def thickness(self, t: u.Quantity) -> None:
        """
        Set the thickness of the surface
        :param t: New thickness of surface
        :return: None
        """

        # Update private variable
        self._thickness = t

        # Update thickness of Zemax row if this surface is connected to a system.
        if self.sys is not None:

            # Convert Quantity to float in Zemax lens units
            t_value = t.to(self.sys.lens_units).value

            # If there is a thickness row defined, update the value from that row
            if self.thickness_str in self._attr_rows:
                self._attr_rows[self.thickness_str].Thickness = t_value

            # Otherwise, update the thickness value from the main row.
            else:
                self._attr_rows_list[-1][1].Thickness = t_value

    @property
    def tilt_z(self) -> u.Quantity:
        return Surface.tilt_z.fget()

    @tilt_z.setter
    def tilt_z(self, val: u.Quantity):

        bcs = self.before_surf_cs_break
        acs = self.after_surf_cs_break

        bcs.R_z = val
        acs.R_z = -val

        self.before_surf_cs_break = bcs
        self.after_surf_cs_break = acs

    @property
    def before_surf_cs_break(self) -> CoordinateSystem:
        """
        CoordinateSystem representing a coordinate break before the optical surface.

        :return: A coordinate break to be applied before the surface.
        """

        # If the value of the coordinate break has not been read from Zemax
        if self._before_surf_cs_break is None:

            # If a cs_break row is defined, convert to a CoordinateSystem object and return.
            if self.before_surf_cs_break_str in self._attr_rows:

                # Find the Zemax LDE row corresponding to the coordinate break for this surface
                row = self._attr_rows[self.before_surf_cs_break_str]

                d1 = self._ISurfaceCoordinateBreak_data(row)

                # Convert a Zemax coordinate break into a coordinate system
                cs1 = self._cs_from_tait_bryant(d1.Decenter_X, d1.Decenter_Y, d1.TiltAbout_X, d1.TiltAbout_Y,
                                                d1.TiltAbout_Z, d1.Order)

            # Otherwise just return a identity coordinate system
            else:
                cs1 = gcs()

            # Add the tilt/decenter within the main surface
            d2 = self._attr_rows[self.main_str].TiltDecenterData
            cs2 = self._cs_from_tait_bryant(d2.BeforeSurfaceDecenterX, d2.BeforeSurfaceDecenterY, d2.BeforeSurfaceTiltX,
                                            d2.BeforeSurfaceTiltY, d2.BeforeSurfaceTiltZ, d2.BeforeSurfaceOrder)

            # The total coordinate break is the composition of the two coordinate systems
            self._before_surf_cs_break = cs1 @ cs2

        return self._before_surf_cs_break

    @before_surf_cs_break.setter
    def before_surf_cs_break(self, cs: CoordinateSystem) -> None:
        """
        Set the coordinate break before the surface

        :param cs: Coordinate system describing the coordinate break
        :return: None
        """

        # Update private variable
        self._before_surf_cs_break = cs

        # Update coordinate break row if we're connected to a optics system
        if self.sys is not None:

            # Grab pointer to tilt/dec data for convenience
            d1 = self._attr_rows[self.main_str].TiltDecenterData

            # Convert CoordinateSystem to Zemax parameters
            X_x, X_y, R_x, R_y, R_z, order = self._cs_to_tait_bryant(cs)

            # If there is already a coordinate break row defined,
            if self.before_surf_cs_break_str in self._attr_rows:

                # Grab pointer to coordinate break row data
                d2 = self._ISurfaceCoordinateBreak_data(self._attr_rows[self.before_surf_cs_break_str])

                # Update coordinate break data
                d2.Decenter_X = X_x
                d2.Decenter_Y = X_y
                d2.TiltAbout_X = R_x
                d2.TiltAbout_Y = R_y
                d2.TiltAbout_Z = R_z

                # Update tilt/dec data
                d1.BeforeSurfaceDecenterX = 0.0
                d1.BeforeSurfaceDecenterY = 0.0
                d1.BeforeSurfaceTiltX = 0.0
                d1.BeforeSurfaceTiltY = 0.0
                d1.BeforeSurfaceTiltZ = 0.0
                d1.BeforeSurfaceOrder = 0

            # Otherwise, there is no coordinate break row
            else:

                # Update tilt/dec data
                d1.BeforeSurfaceDecenterX = X_x
                d1.BeforeSurfaceDecenterY = X_y
                d1.BeforeSurfaceTiltX = R_x
                d1.BeforeSurfaceTiltY = R_y
                d1.BeforeSurfaceTiltZ = R_z
                d1.BeforeSurfaceOrder = order

    @property
    def after_surf_cs_break(self) -> CoordinateSystem:
        """
        CoordinateSystem representing a coordinate break after the optical surface.

        :return: A coordinate break to be applied after the surface.
        """

        # If the value of the coordinate break has not been read from Zemax
        if self._after_surf_cs_break is None:

            # If a cs_break row is defined, convert to a CoordinateSystem object and return.
            if self.after_surf_cs_break_str in self._attr_rows:

                # Find the Zemax LDE row corresponding to the coordinate break for this surface
                row = self._attr_rows[self.after_surf_cs_break_str]

                d1 = self._ISurfaceCoordinateBreak_data(row)

                # Convert a Zemax coordinate break into a coordinate system
                cs1 = self._cs_from_tait_bryant(d1.Decenter_X, d1.Decenter_Y, d1.TiltAbout_X, d1.TiltAbout_Y,
                                                d1.TiltAbout_Z, d1.Order)

            # Otherwise just return a identity coordinate system
            else:
                cs1 = gcs()

            # Add the tilt/decenter within the main surface
            d2 = self._attr_rows[self.main_str].TiltDecenterData
            cs2 = self._cs_from_tait_bryant(d2.AfterSurfaceDecenterX, d2.AfterSurfaceDecenterY, d2.AfterSurfaceTiltX,
                                            d2.AfterSurfaceTiltY, d2.AfterSurfaceTiltZ, d2.AfterSurfaceOrder)

            # The total coordinate break is the composition of the two coordinate systems
            self._after_surf_cs_break = cs1 @ cs2

        return self._after_surf_cs_break

    @after_surf_cs_break.setter
    def after_surf_cs_break(self, cs: CoordinateSystem) -> None:
        """
        Set the coordinate break after the surface

        :param cs: Coordinate system describing the coordinate break
        :return: None
        """

        # Update private variable
        self._after_surf_cs_break = cs

        # Update coordinate break row if we're connected to a optics system
        if self.sys is not None:

            # Grab pointer to tilt/dec data for convenience
            d1 = self._attr_rows[self.main_str].TiltDecenterData

            # Convert CoordinateSystem to Zemax parameters
            X_x, X_y, R_x, R_y, R_z, order = self._cs_to_tait_bryant(cs)

            # If there is already a coordinate break row defined,
            if self.after_surf_cs_break_str in self._attr_rows:

                # Grab pointer to coordinate break row data
                d2 = self._ISurfaceCoordinateBreak_data(self._attr_rows[self.after_surf_cs_break_str])

                # Update coordinate break data
                d2.Decenter_X = X_x
                d2.Decenter_Y = X_y
                d2.TiltAbout_X = R_x
                d2.TiltAbout_Y = R_y
                d2.TiltAbout_Z = R_z

                # Update tilt/dec data
                d1.AfterSurfaceDecenterX = 0.0
                d1.AfterSurfaceDecenterY = 0.0
                d1.AfterSurfaceTiltX = 0.0
                d1.AfterSurfaceTiltY = 0.0
                d1.AfterSurfaceTiltZ = 0.0
                d1.AfterSurfaceOrder = 0

            # Otherwise, there is no coordinate break row
            else:

                # Update tilt/dec data
                d1.AfterSurfaceDecenterX = X_x
                d1.AfterSurfaceDecenterY = X_y
                d1.AfterSurfaceTiltX = R_x
                d1.AfterSurfaceTiltY = R_y
                d1.AfterSurfaceTiltZ = R_z
                d1.AfterSurfaceOrder = order

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
    def _attr_rows(self) -> 'Dict[str, ILDERow]':
        """
        The attr_rows are stored internally as a list, but need to be accessed as a dict
        :return: A dict of attributes, where the attribute name is the key and the Zemax row is the value.
        """
        ret = OrderedDict()
        for name, row in self._attr_rows_list:
            ret[name] = row

        return ret

    @_attr_rows.setter
    def _attr_rows(self, attr_rows: 'Dict[str, ILDERow]') -> None:
        """
        Write to the ordered dict of attributes, checking that all the Zemax rows are consecutive and that the surfaces
        are ordered in the correct priority
        :param attr_rows: Ordered dict of attributes, where the key is the attribute name and the value is a Zemax row.
        :return: None
        """

        # Initial vales for the last index and last priority
        last_ind = None     # type: Union[int, None]
        last_pri = -1

        # Space to store result of converting dict to list
        attr_rows_list = []         # type: List[Tuple[str, ILDERow]]

        # Loop through all the attributes in the dictionary and check that each element is consecutive and in ascending
        # priority, and add to list of surfaces
        for attr, row in attr_rows.items():

            # Extract the index and priority values for comparison
            ind = row.SurfaceNumber
            pri = self.priority_dict[attr]

            # Check that this index is one greater than the previous index
            if last_ind is not None:
                if ind != last_ind + 1:
                    raise ValueError('Non-consecutive attribute rows')

            # Check that this priority is greater than or equal to the previous priority
            if pri < last_pri:
                raise ValueError('Higher priority surface before lower priority surface')

            # Add attribute to list
            attr_rows_list.append((attr, row))

            # Update persistent variable to compare at the next loop iteration
            last_ind = ind
            last_pri = pri

        # Update private storage variable
        self._attr_rows_list = attr_rows_list

    def insert(self, attr_name: str, row_type: ZOSAPI.Editors.LDE.SurfaceType) -> None:
        """
        Insert a new attribute row, with it's position specified by the priority of the attribute
        :param attr_name: Name of the attribute, determines priority
        :param row_type: The type of ILDErow that should be associated with the attribute
        :return: None
        """

        # Initialize the index variable for the case when there are no attribute rows in the list
        i = 0

        # Initialize the row index variable so we know if its been set
        row_ind = None

        # If there is at least one row in the surface, locate where the new row should go based on priority
        if self._attr_rows_list:

            # Check that the attribute doesn't already have an associated Zemax row
            if attr_name in self._attr_rows:
                raise ValueError('Attribute already defined')

            # Extract the priority of the provided attribute
            pri = self.priority_dict[attr_name]

            # Loop through all the attributes, find where the new attribute should go, and insert the new attribute
            for i, item in enumerate(self._attr_rows_list):

                # Extract name and ILDE row from tuple
                attr, row = item

                # If the priority of the provided attribute is less than the current row, this is the correct index for
                # the new attribute row.
                if pri < self.priority_dict[attr]:

                    # Insert a new row at the index of the current row
                    row_ind = row.RowIndex

                    # Stop the loop so we don't keep adding surfaces
                    break

            # If the row index was not set in the previous loop, set it to the last element in the loop
            if row_ind is None:
                row_ind = self.last_row_ind + 1

        # Otherwise, there are no rows in the surface and we have to locate our position based off of the previous
        # surface.
        else:

            # Insert a new row at the index after the last row in the previous surface
            row_ind = self.prev_surf_in_system.last_row_ind + 1

        # Create a new Zemax row based off of the index
        new_row = self.sys.zos_sys.LDE.InsertNewSurfaceAt(row_ind)

        # Change the type of the row to the requested type
        new_row.ChangeType(new_row.GetSurfaceTypeSettings(row_type))

        # Update the row with the correct control comment.
        new_row.Comment = self.component.name + '.' + self.name + '.' + attr_name

        # Add the new row to the attribute list
        self._attr_rows_list.insert(i, (attr_name, new_row))

        # Update the attributes of the surface
        self.aperture = self.aperture
        self.thickness = self.thickness
        self.before_surf_cs_break = self.before_surf_cs_break
        self.after_surf_cs_break = self.after_surf_cs_break

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
            indices.append(row.SurfaceNumber)

        return indices

    @property
    def first_row_ind(self) -> Union[int, None]:
        """
        :return: The index of the first row of this surface in the Zemax file if it exists, otherwise return None.
        """
        if self._attr_rows_indices:
            return self._attr_rows_indices[0]
        else:
            return None

    @property
    def last_row_ind(self) -> Union[int, None]:
        """
        :return: The index of the last row of this surface in the Zemax file if it exists, otherwise return None.
        """
        if self._attr_rows_indices:
            return self._attr_rows_indices[-1]
        else:
            return None

    def _cs_from_tait_bryant(self, X_x: float, X_y: float, R_x: float, R_y: float, R_z: float,
                             order: int):
        """
        Construct a Coordinate System using Zemax Tilt/Decenter parameters.

        :param X_x: Decenter x
        :param X_y: Decenter y
        :param R_x: Tilt about x
        :param R_y: Tilt about y
        :param R_z: Tilt about z
        :param order: If zero, the decenter is first, otherwise the tilt is first.
        :return:
        """

        # Express angles as a degree quantity
        R_x = R_x * u.deg       # type: u.Quantity
        R_y = R_y * u.deg       # type: u.Quantity
        R_z = R_z * u.deg       # type: u.Quantity

        # Convert to radians
        R_x = R_x.to(u.rad)
        R_y = R_y.to(u.rad)
        R_z = R_z.to(u.rad)

        # Construct vector from decenter parameters
        X = Vector([X_x, X_y, 0] * self.sys.lens_units)

        # Construct rotation quaternion from tilt parameters
        Q = from_xyz_intrinsic_tait_bryan_angles(R_x, R_y, R_z)

        # Determine which order the tilts/decenters should be executed
        if order == 0:
            translation_first = True
        else:
            translation_first = False

        # Create new coordinate system representing the Zemax coordinate break
        cs = CoordinateSystem(X, Q, translation_first=translation_first)

        return cs

    def _cs_to_tait_bryant(self, cs: CoordinateSystem) -> Tuple[float, float, float, float, float, int]:
        """
        Construct Zemax tilt/decenter parameters from a coordinate system

        :param cs: Coordinate system to convert
        :return: Decenter x, decenter y, tilt x, tilt y, tilt z, order
        """

        # Check that we have a coordinate system that is compatible with a coordinate break
        if cs.X.z != 0:
            raise ValueError('Nonzero z-translation')

        # Update the order
        if cs.translation_first:
            order = 0
        else:
            order = 1

        # Update the translation
        X_x = cs.X.x.to(self.sys.lens_units).value
        X_y = cs.X.y.to(self.sys.lens_units).value

        # Update the rotation
        R_x, R_y, R_z = as_xyz_intrinsic_tait_bryan_angles(cs.Q)

        # Express angles as a radian quantity
        R_x = R_x * u.rad       # type: u.Quantity
        R_y = R_y * u.rad       # type: u.Quantity
        R_z = R_z * u.rad       # type: u.Quantity

        # Convert to degrees
        R_x = R_x.to(u.deg).value
        R_y = R_y.to(u.deg).value
        R_z = R_z.to(u.deg).value

        return X_x, X_y, R_x, R_y, R_z, order

    def __getitem__(self, item: Union[int, str]) -> ILDERow:
        """
        Gets the ILDERow at index item within the surface, or the ILDERow with the name item
        :param item: Surface index or name of surface
        :return: ILDERow specified by item
        """

        # If the item is an integer, use it to access the attribute row list
        # Note that the row is located at index 1 in the tuple
        if isinstance(item, int):
            return self._attr_rows_list.__getitem__(item).__getitem__(1)

        # If the item is a string, use it to access the surfaces dictionary.
        elif isinstance(item, str):
            return self._attr_rows.__getitem__(item)

        # Otherwise, the item is neither an int nor string and we throw an error.
        else:
            raise ValueError('Item is of an unrecognized type')

    def __setitem__(self, key: Union[int, str], value: ILDERow):

        # If the key is an int, set the item at the index
        if isinstance(key, int):

            # Make copy of current contents
            old_val = self._attr_rows_list.__getitem__(key)

            # Make new tuple, with new value for row
            new_val = (old_val[0], value)

            # Set the list element with our new value
            self._attr_rows_list.__setitem__(key, new_val)

        # If the item is a string, find the index of the value and call this function again
        elif isinstance(key, str):

            # Find instance of attribute row using key
            attr = self[key]

            # Find index of this attribute row in the list
            ind = self._attr_rows_list.index((key, attr))

            # Recursively call this function to set using index
            self.__setitem__(ind, value)

        # Otherwise, the item is neither an int nor string and we throw an error.
        else:
            raise ValueError('Item is of an unrecognized type')

    def __delitem__(self, key: int):

        self._attr_rows_list.__delitem__(key)

    def __iter__(self) -> Iterator[Tuple[str,ILDERow]]:

        return self._attr_rows_list.__iter__()

    def __len__(self):

        return self._attr_rows_list.__len__()

    # def __del__(self) -> None:
    #     """
    #     Delete method for the surface.
    #     Removes all ILDERows associated with the surface from the LDE
    #     :return: None
    #     """
    #
    #     rows = self.attr_rows.copy()
    #
    #     for attr, row in rows.items():
    #
    #         print('---------------')
    #         print(self.sys.zos_sys.LDE)
    #         self.sys.zos_sys.LDE.RemoveSurfaceAt(row.SurfaceNumber)
    #         print(row.SurfaceData)
    #
    #         self._attr_rows_var.move_to_end(attr,)
    #
    #         del self._attr_rows_var[attr]














