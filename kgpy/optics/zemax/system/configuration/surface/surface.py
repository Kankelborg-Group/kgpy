
import typing as tp
from win32com.client import CastTo
from typing import List, Dict, Union, Tuple, Iterator
import numpy as np
import astropy.units as u
from collections import OrderedDict
from enum import IntEnum, auto

from kgpy import optics

import kgpy.optics
from kgpy.math.quaternion import *
from kgpy.math import Vector, CoordinateSystem
from kgpy.math.coordinate_system import GlobalCoordinateSystem as gcs
from kgpy.optics.system.configuration import surface
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system.configuration.surface import Material
from kgpy.optics.zemax.ZOSAPI.Editors.LDE import ILDERow

__all__ = ['Surface']


class Surface(optics.system.configuration.Surface):
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
    aperture_str = 'aper'
    mech_aper_str = 'mech_aper'

    class AttrPriority(IntEnum):
        """
        Enumeration class to describe the priority that each attribute gets when the surface is written to file.
        """

        before_surf_cs_break = auto()
        aper = auto()
        mech_aper = auto()
        main = auto()
        after_surf_cs_break = auto()
        thickness = auto()

    # Dictionary that maps the attribute strings to their priority level
    priority_dict = {
        main_str:                   AttrPriority.main,
        before_surf_cs_break_str:   AttrPriority.before_surf_cs_break,
        after_surf_cs_break_str:    AttrPriority.after_surf_cs_break,
        thickness_str:              AttrPriority.thickness,
        aperture_str:                   AttrPriority.aper,
        mech_aper_str:              AttrPriority.mech_aper
    }

    def __init__(self, name: str):
        """
        Constructor for the ZmxSurface class.
        This constructor places the surface at the origin of the global coordinate system, it needs to be moved into
        place after the call to this function.
        :param name: Human-readable name of the surface
        """

        # Initialize list of ILDERows associated with class attributes.
        # This needs to be before the superclass constructor so properties such as thickness are defined
        self._attr_rows = {}

        # Call superclass constructor
        Surface.__init__(self, name)

        # Override the type of the system pointer
        self.configuration = None

        # Override the type of the neighboring surfaces
        self.prev_surf_in_system = None         # type: Surface
        self.next_surf_in_system = None         # type: Surface
        self.prev_surf_in_component = None      # type: Surface
        self.next_surf_in_component = None      # type: Surface

        self.aperture = None
        self.mechanical_aperture = None
        self.material = None
        self.surface_type = None

        self.radius = np.inf * u.mm
        self.conic = 0.0

        self.explicit_csb = False

    @property
    def configuration(self) -> tp.Union[None, 'optics.zemax.system.Configuration']:
        return self._configuration

    @configuration.setter
    def configuration(self, value: tp.Union[None, 'optics.zemax.system.Configuration']):
        self._configuration = value

    def _prep_mce(self):

        self._after_decenter_x_op = self.system.zos_sys.MCE.AddOperand()
        self._after_decenter_x_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.CADX)
        self._after_decenter_x_op.Param1 = self.main_row.SurfaceNumber

        self._after_decenter_y_op = self.system.zos_sys.MCE.AddOperand()
        self._after_decenter_y_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.CADY)
        self._after_decenter_y_op.Param1 = self.main_row.SurfaceNumber

        self._after_tilt_x_op = self.system.zos_sys.MCE.AddOperand()
        self._after_tilt_x_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.CATX)
        self._after_tilt_x_op.Param1 = self.main_row.SurfaceNumber

        self._after_tilt_y_op = self.system.zos_sys.MCE.AddOperand()
        self._after_tilt_y_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.CATY)
        self._after_tilt_y_op.Param1 = self.main_row.SurfaceNumber

        self._after_tilt_z_op = self.system.zos_sys.MCE.AddOperand()
        self._after_tilt_z_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.CATZ)
        self._after_tilt_z_op.Param1 = self.main_row.SurfaceNumber

        self._after_order_op = self.system.zos_sys.MCE.AddOperand()
        self._after_order_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.CAOR)
        self._after_order_op.Param1 = self.main_row.SurfaceNumber

        self._before_decenter_x_op = self.system.zos_sys.MCE.AddOperand()
        self._before_decenter_x_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.CBDX)
        self._before_decenter_x_op.Param1 = self.main_row.SurfaceNumber

        self._before_decenter_y_op = self.system.zos_sys.MCE.AddOperand()
        self._before_decenter_y_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.CBDY)
        self._before_decenter_y_op.Param1 = self.main_row.SurfaceNumber

        self._before_tilt_x_op = self.system.zos_sys.MCE.AddOperand()
        self._before_tilt_x_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.CBTX)
        self._before_tilt_x_op.Param1 = self.main_row.SurfaceNumber

        self._before_tilt_y_op = self.system.zos_sys.MCE.AddOperand()
        self._before_tilt_y_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.CBTY)
        self._before_tilt_y_op.Param1 = self.main_row.SurfaceNumber

        self._before_tilt_z_op = self.system.zos_sys.MCE.AddOperand()
        self._before_tilt_z_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.CBTZ)
        self._before_tilt_z_op.Param1 = self.main_row.SurfaceNumber

        self._before_order_op = self.system.zos_sys.MCE.AddOperand()
        self._before_order_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.CBOR)
        self._before_order_op.Param1 = self.main_row.SurfaceNumber

    def _prep_mce_coordinate_break(self, row: ILDERow):

        for i in range(1, 7):

            op = self.system.zos_sys.MCE.AddOperand()
            op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM)
            op.Param1 = row.SurfaceNumber
            op.Param2 = i

    @staticmethod
    def conscript(surf: optics.system.configuration.Surface) -> 'Surface':
        """
        Convert a Surface object to ZmxSurface object
        :param surf: Surface object to convert
        :return: a new ZmxSurface object equal to the provided Surface object
        """

        # Construct new ZmxSurface
        zmx_surf = Surface(surf.name)

        zmx_surf.explicit_csb = surf.explicit_csb

        # Copy remaining attributes
        zmx_surf.surface_type = surf.surface_type
        zmx_surf.comment = surf.comment
        zmx_surf.before_surf_cs_break = surf.before_surf_cs_break
        zmx_surf.after_surf_cs_break = surf.after_surf_cs_break
        zmx_surf.component = surf.component
        zmx_surf.aperture = surf.aperture
        zmx_surf.mechanical_aperture = surf.mechanical_aperture
        zmx_surf.radius = surf.radius
        zmx_surf.conic = surf.conic
        zmx_surf.material = surf.material
        zmx_surf.thickness = surf.thickness

        return zmx_surf

    @staticmethod
    def from_attr_dict(surf_name: str, attr_dict: Dict[str, ILDERow]) -> 'Surface':

        # Construct new ZmxSurface
        zmx_surf = Surface(surf_name)

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
    def surface_type(self):
        return self._surface_type
    
    @surface_type.setter
    def surface_type(self, value: kgpy.optics.system.configuration.surface.surface_type.SurfaceType):
        
        if (self.system is not None) and (value is not None):

            self._surface_type = value.promote_to_zmx(self)

        else:

            self._surface_type = value
    
    @property
    def conic(self) -> float:
        return self._conic
    
    @conic.setter
    def conic(self, value: float):
        self._conic = value

        if self.system is not None:
            self.main_row.Conic = self._conic

            if self.aperture_str in self.attr_rows:
                self.attr_rows[self.aperture_str].Conic = self._conic

    
    @property
    def attr_rows(self):
        return self._attr_rows

    @property
    def material(self) -> surface.Material:
        return self._material

    @material.setter
    def material(self, val: surface.Material):
        
        if self.system is not None and val is not None:

            self._material = Material(val.name, self)

        else:

            self._material = val

    @property
    def aperture(self) -> surface.Aperture:
        return self._aperture

    @aperture.setter
    def aperture(self, aper: surface.Aperture):

        if (self.system is not None) and (aper is not None):

            if self.aperture_str not in self.attr_rows:
                self.insert_row(self.aperture_str, ZOSAPI.Editors.LDE.SurfaceType.Standard)

            self._aperture = aper.promote_to_zmx(self, self.aperture_str)

        else:

            self._aperture = aper

    @property
    def mechanical_aperture(self):

        return self._mechanical_aperture

    @mechanical_aperture.setter
    def mechanical_aperture(self, value: surface.Aperture):

        if (self.system is not None) and (value is not None):

            self._mechanical_aperture = value.promote_to_zmx(self, self.main_str)

        else:

            self._mechanical_aperture = value


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
            self._radius = self._attr_rows[self.main_str].Radius * self.system.lens_units

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
        # noinspection PyPep8Naming

        if self.system is not None:

            self._attr_rows[self.main_str].Radius = val.to(self.system.lens_units).value

            if self.aperture_str in self.attr_rows:
                self.attr_rows[self.aperture_str].Radius = val.to(self.system.lens_units).value

    @property
    def is_stop(self) -> bool:
        """
        :return: True if this surface contains a stop row, False otherwise
        """

        # If the surface is part of a ZOS system, return the stop flag of the main surface
        if self.system is not None:

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
        # noinspection PyPep8Naming
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
                self._thickness = self._attr_rows[self.thickness_str].Thickness * self.system.lens_units

            # Otherwise, return the thickness value from the last row in the surface
            else:
                self._thickness = self._attr_rows_list[-1][1].Thickness * self.system.lens_units

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
        if self.system is not None:

            # Convert Quantity to float in Zemax lens units
            t_value = t.to(self.system.lens_units).value

            if self.explicit_csb or len(self.attr_rows) > 1:

                if self.thickness_str not in self._attr_rows:
                    self.insert_row(self.thickness_str, ZOSAPI.Editors.LDE.SurfaceType.Standard)

                self._attr_rows[self.thickness_str].Thickness = t_value

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

        :return: a coordinate break to be applied before the surface.
        """

        # If the value of the coordinate break has not been read from Zemax
        if self._before_surf_cs_break_list[self.config] is None:

            # If a cs_break row is defined, convert to a CoordinateSystem object and return.
            if self.before_surf_cs_break_str in self._attr_rows:

                # Find the Zemax LDE row corresponding to the coordinate break for this surface
                row = self._attr_rows[self.before_surf_cs_break_str]

                d1 = self._ISurfaceCoordinateBreak_data(row)

                # Convert a Zemax coordinate break into a coordinate system
                cs1 = self._cs_from_tait_bryant(d1.Decenter_X, d1.Decenter_Y, row.Thickness, d1.TiltAbout_X,
                                                d1.TiltAbout_Y, d1.TiltAbout_Z, d1.Order)

            # Otherwise just return a identity coordinate system
            else:
                cs1 = gcs()

            # Add the tilt/decenter within the main surface
            d2 = self._attr_rows[self.main_str].TiltDecenterData
            cs2 = self._cs_from_tait_bryant(d2.BeforeSurfaceDecenterX, d2.BeforeSurfaceDecenterY, 0.0,
                                            d2.BeforeSurfaceTiltX, d2.BeforeSurfaceTiltY, d2.BeforeSurfaceTiltZ,
                                            d2.BeforeSurfaceOrder)

            # The total coordinate break is the composition of the two coordinate systems
            self._before_surf_cs_break_list[self.config] = cs1 @ cs2

        return self._before_surf_cs_break_list[self.config]

    @before_surf_cs_break.setter
    def before_surf_cs_break(self, cs: CoordinateSystem) -> None:
        """
        Set the coordinate break before the surface

        :param cs: Coordinate system describing the coordinate break
        :return: None
        """

        # Update private variable
        self._before_surf_cs_break_list[self.config] = cs

        # Update coordinate break row if we're connected to a optics system
        if self.system is not None:

            if self.explicit_csb or len(self.attr_rows) > 1:
                if self.before_surf_cs_break_str not in self._attr_rows:
                    self.insert_row(self.before_surf_cs_break_str, ZOSAPI.Editors.LDE.SurfaceType.CoordinateBreak)
                    self._prep_mce_coordinate_break(self._attr_rows[self.before_surf_cs_break_str])

            # Grab pointer to tilt/dec data for convenience
            d1 = self._attr_rows[self.main_str].TiltDecenterData

            # Convert CoordinateSystem to Zemax parameters
            X_x, X_y, X_z, R_x, R_y, R_z, order = self._cs_to_tait_bryant(cs)

            # If there is already a coordinate break row defined,
            if self.before_surf_cs_break_str in self._attr_rows:

                cb_row = self._attr_rows[self.before_surf_cs_break_str]

                cb_row.Thickness = X_z

                # Grab pointer to coordinate break row data
                d2 = self._ISurfaceCoordinateBreak_data(cb_row)

                # Update coordinate break data
                d2.Decenter_X = X_x
                d2.Decenter_Y = X_y
                d2.TiltAbout_X = R_x
                d2.TiltAbout_Y = R_y
                d2.TiltAbout_Z = R_z
                d2.Order = order

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

        :return: a coordinate break to be applied after the surface.
        """

        # If the value of the coordinate break has not been read from Zemax
        if self._after_surf_cs_break_list[self.config] is None:

            # If a cs_break row is defined, convert to a CoordinateSystem object and return.
            if self.after_surf_cs_break_str in self._attr_rows:

                # Find the Zemax LDE row corresponding to the coordinate break for this surface
                row = self._attr_rows[self.after_surf_cs_break_str]

                d1 = self._ISurfaceCoordinateBreak_data(row)

                # Convert a Zemax coordinate break into a coordinate system
                cs1 = self._cs_from_tait_bryant(d1.Decenter_X, d1.Decenter_Y, row.Thickness, d1.TiltAbout_X, d1.TiltAbout_Y,
                                                d1.TiltAbout_Z, d1.Order)

            # Otherwise just return a identity coordinate system
            else:
                cs1 = gcs()

            # Add the tilt/decenter within the main surface
            d2 = self._attr_rows[self.main_str].TiltDecenterData
            cs2 = self._cs_from_tait_bryant(d2.AfterSurfaceDecenterX, d2.AfterSurfaceDecenterY, 0.0, d2.AfterSurfaceTiltX,
                                            d2.AfterSurfaceTiltY, d2.AfterSurfaceTiltZ, d2.AfterSurfaceOrder)

            # The total coordinate break is the composition of the two coordinate systems
            self._after_surf_cs_break_list[self.config] = cs1 @ cs2

        return self._after_surf_cs_break_list[self.config]

    @after_surf_cs_break.setter
    def after_surf_cs_break(self, cs: CoordinateSystem) -> None:
        """
        Set the coordinate break after the surface

        :param cs: Coordinate system describing the coordinate break
        :return: None
        """

        # Update private variable
        self._after_surf_cs_break_list[self.config] = cs

        # Update coordinate break row if we're connected to a optics system
        if self.system is not None and not self.main_row.IsImage:

            if self.explicit_csb or len(self.attr_rows) > 1:
                if self.after_surf_cs_break_str not in self._attr_rows:
                    self.insert_row(self.after_surf_cs_break_str, ZOSAPI.Editors.LDE.SurfaceType.CoordinateBreak)
                    self._prep_mce_coordinate_break(self._attr_rows[self.after_surf_cs_break_str])

            # Grab pointer to tilt/dec data for convenience
            d1 = self._attr_rows[self.main_str].TiltDecenterData

            # Convert CoordinateSystem to Zemax parameters
            X_x, X_y, X_z, R_x, R_y, R_z, order = self._cs_to_tait_bryant(cs)

            # If there is already a coordinate break row defined,
            if self.after_surf_cs_break_str in self._attr_rows:

                cb_row = self._attr_rows[self.after_surf_cs_break_str]

                cb_row.Thickness = X_z

                # Grab pointer to coordinate break row data
                d2 = self._ISurfaceCoordinateBreak_data(cb_row)

                # Update coordinate break data
                d2.Decenter_X = X_x
                d2.Decenter_Y = X_y
                d2.TiltAbout_X = R_x
                d2.TiltAbout_Y = R_y
                d2.TiltAbout_Z = R_z
                d2.Order = order

                # Update tilt/dec data
                d1.AfterSurfaceDecenterX = 0.0
                d1.AfterSurfaceDecenterY = 0.0
                d1.AfterSurfaceTiltX = 0.0
                d1.AfterSurfaceTiltY = 0.0
                d1.AfterSurfaceTiltZ = 0.0
                d1.AfterSurfaceOrder = 1

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
        :return: a dict of attributes, where the attribute name is the key and the Zemax row is the value.
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

        self.insert_row(attr_name, row_type)

        self._prep_mce()

        self.update()

    def update(self):

        # Update the attributes of the surface
        self.surface_type = self.surface_type
        self.aperture = self.aperture
        self.before_surf_cs_break = self.before_surf_cs_break
        self.after_surf_cs_break = self.after_surf_cs_break
        self.mechanical_aperture = self.mechanical_aperture
        self.radius = self.radius
        self.conic = self.conic
        self.material = self.material
        self.thickness = self.thickness

        self.explicit_csb = self.explicit_csb

    def insert_row(self, attr_name: str, row_type: ZOSAPI.Editors.LDE.SurfaceType):

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
                    row_ind = row.SurfaceNumber

                    # Stop the loop so we don't keep adding surfaces
                    break

            # If the row index was not set in the previous loop, set it to the last element in the loop
            if row_ind is None:
                i += 1
                row_ind = self.last_row_ind + 1

        # Otherwise, there are no rows in the surface and we have to locate our position based off of the previous
        # surface.
        else:

            # Insert a new row at the index after the last row in the previous surface
            row_ind = self.prev_surf_in_system.last_row_ind + 1

        # Create a new Zemax row based off of the index
        new_row = self.system.zos_sys.LDE.InsertNewSurfaceAt(row_ind)

        # Change the type of the row to the requested type
        new_row.ChangeType(new_row.GetSurfaceTypeSettings(row_type))

        if self.component is not None:
            comp_name = self.component.name
        else:
            comp_name = 'Main'

        # Update the row with the correct control comment.
        new_row.Comment = comp_name + '.' + self.name + '.' + attr_name

        # Add the new row to the attribute list
        self._attr_rows_list.insert(i, (attr_name, new_row))

        self.system._update_system_from_zmx()

    def raytrace(self, configurations: List[int], surfaces: List[Surface], wavelengths: List[wavelength.Item],
                 field_x: np.ndarray, field_y: np.ndarray, pupil_x: np.ndarray, pupil_y: np.ndarray,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # Grab a handle to the zemax system
        sys = self.zos_sys

        # Store number of surfaces
        num_surf = len(surfaces)

        # store number of wavelengths
        num_wavl = len(wavelengths)

        # Store length of each axis in ray grid
        num_field_x = len(field_x)
        num_field_y = len(field_y)
        num_pupil_x = len(pupil_x)
        num_pupil_y = len(pupil_y)

        # Create grid of rays
        Fx, Fy, Px, Py = np.meshgrid(field_x, field_y, pupil_x, pupil_y, indexing='ij')

        # Store shape of grid
        sh = list(Fx.shape)

        # Shape of grid for each surface and wavelength
        tot_sh = [self.num_configurations, num_surf, num_wavl] + sh

        # Allocate output arrays
        V = np.empty(tot_sh)  # Vignetted rays
        X = np.empty(tot_sh)
        Y = np.empty(tot_sh)

        old_config = self.config

        for c in configurations:

            self.config = c

            # Initialize raytrace
            rt = self.zos_sys.Tools.OpenBatchRayTrace()  # raytrace object
            tool = sys.Tools.CurrentTool  # pointer to active tool

            # Loop over each surface and run raytrace to surface
            for s, surf in enumerate(surfaces):

                # Open instance of batch raytrace
                rt_dat = rt.CreateNormUnpol(num_pupil_x * num_pupil_y, constants.RaysType_Real,
                                            surf.main_row.SurfaceNumber)

                # Run raytrace for each wavelength
                for w, wavl in enumerate(wavelengths):

                    # Run raytrace for each field angle
                    for fi in range(num_field_x):
                        for fj in range(num_field_y):

                            rt_dat.ClearData()

                            # Loop over pupil to add rays to batch raytrace
                            for pi in range(num_pupil_x):
                                for pj in range(num_pupil_y):
                                    # Select next ray
                                    fx = Fx[fi, fj, pi, pj]
                                    fy = Fy[fi, fj, pi, pj]
                                    px = Px[fi, fj, pi, pj]
                                    py = Py[fi, fj, pi, pj]

                                    # Write ray to pipe
                                    rt_dat.AddRay(wavl.zos_wavl.WavelengthNumber, fx, fy, px, py,
                                                  constants.OPDMode_None)

                            # Execute the raytrace
                            tool.RunAndWaitForCompletion()

                            # Initialize the process of reading the results of the raytrace
                            rt_dat.StartReadingResults()

                            # Loop over pupil and read the results of the raytrace
                            for pi in range(num_pupil_x):
                                for pj in range(num_pupil_y):
                                    # Read next result from pipe
                                    (ret, n, err, vig, x, y, z, l, m, n, l2, m2, n2, opd,
                                     I) = rt_dat.ReadNextResult()

                                    # Store next result in output arrays
                                    V[c, s, w, fi, fj, pi, pj] = vig
                                    X[c, s, w, fi, fj, pi, pj] = x
                                    Y[c, s, w, fi, fj, pi, pj] = y

            tool.Close()

        self.config = old_config

        return V, X, Y

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

    def _cs_from_tait_bryant(self, X_x: float, X_y: float, X_z: float, R_x: float, R_y: float, R_z: float,
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
        X = Vector([X_x, X_y, X_z] * self.system.lens_units)

        # Construct rotation quaternion from tilt parameters
        Q = from_xyz_intrinsic_tait_bryan_angles(R_x, R_y, R_z, x_first=False)

        # Create new coordinate system representing the Zemax coordinate break
        cs = CoordinateSystem(X, Q, translation_first=False)

        return cs

    def _cs_to_tait_bryant(self, cs: CoordinateSystem) -> Tuple[float, float, float, float, float, float, int]:
        """
        Construct Zemax tilt/decenter parameters from a coordinate system

        :param cs: Coordinate system to convert
        :return: Decenter x, decenter y, tilt x, tilt y, tilt z, order
        """

        order = 1

        X = cs.X.rotate(cs.Q.inverse())

        # Update the translation
        X_x = X.x.to(self.system.lens_units).value
        X_y = X.y.to(self.system.lens_units).value
        X_z = X.z.to(self.system.lens_units).value

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

        return X_x, X_y, X_z, R_x, R_y, R_z, order

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














