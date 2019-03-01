
from win32com.client import CastTo
from typing import List, Dict
import astropy.units as u

from kgpy.math.quaternion import *
from kgpy.math import Vector, CoordinateSystem
from kgpy.math.coordinate_system import GlobalCoordinateSystem as gcs
from kgpy.optics import Surface, Component
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.ZOSAPI.Editors.LDE import ILDERow

__all__ = ['ZmxSurface']


class ZmxSurface(Surface):
    """
    Child of the Surface class which acts as a wrapper around the Zemax API ILDERow class (a Zemax surface).
    The ZmxSurface class is intended to be used temporarily as the functionality in the ILDERow class is implemented in
    the Surface superclass.
    """

    # noinspection PyMissingConstructor
    def __init__(self, name: str, row: ILDERow, length_units: u.Unit):
        """
        Constructor for ZmxSurface object.
        :param row: Pointer to the Zemax ILDERow to wrap this class around
        """

        # Save arguments to class variables
        self.name = name
        self.row = row
        self.u = length_units

        # Attributes to be set by Component.append_surface()
        self.prev_surf_in_system = None        # type: Surface
        self.next_surf_in_system = None        # type: Surface
        self.prev_surf_in_component = None     # type: Surface
        self.next_surf_in_component = None     # type: Surface
        self.component = None                  # type: Component

        # Initialize class variables
        self.attr_rows = {}        # type: Dict[str, ILDERow]

    def _get_attr_row(self, attr: str) -> ILDERow:
        """
        Finds the row that corresponds to a particular attribute.
        In this system, several rows in the Zemax LDE can contribute to a single surface in this system.
        This function finds the row responsible for tracking an attribute.
        :param attr: Name of the attribute
        :return: Row corresponding to that attribute
        """

        # If the attribute is a key in the dictionary, return the row corresponding to that key.
        if attr in self.attr_rows:
            return self.attr_rows[attr]

        # Otherwise return the main row
        else:
            return self.row

    @property
    def thickness(self) -> u.Quantity:
        """
        :return: Thickness of the surface in lens units
        """

        return self._get_attr_row('thickness').Thickness * self.u

    @thickness.setter
    def thickness(self, t: u.Quantity) -> None:
        """
        Set the thickness of the surface
        :param t: New thickness of surface
        :return: None
        """

        self._get_attr_row('thickness').Thickness = float(t / self.u)

    @property
    def cs_break(self) -> CoordinateSystem:
        """
        Compute the coordinate system for this coordinate break.
        Zemax represents the tilts of a optical surface using the xyz intrinsic Tait-Bryan angles.
        This function tries to find a coordinate break surface within the list of attribute rows, and if it exists,
        constructs a new coordinate system to represent the coordinate break.
        :return: A coordinate system representing a Zemax coordinate break.
        """

        # Find the Zemax LDE row corresponding to the coordinate break for this surface
        row = self._get_attr_row('cs_break')

        # If the row is a CoordinateBreak surface, construct a new coordinate system to represent the new coordinate
        # system.
        if row.Type == ZOSAPI.Editors.LDE.SurfaceType.CoordinateBreak:

            # Store a pointer to coordinate break parameters
            data = CastTo(row.SurfaceData, ZOSAPI.Editors.LDE.
                          ISurfaceCoordinateBreak.__name__)     # type: ZOSAPI.Editors.LDE.ISurfaceCoordinateBreak

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

        # Find the Zemax LDE row corresponding to the coordinate break for this surface
        row = self._get_attr_row('cs_break')

        # If the row is a CoordinateBreak surface, update the parameters with the new value from the provided coordinate
        # system.
        if row.Type == ZOSAPI.Editors.LDE.SurfaceType.CoordinateBreak:

            # Store a pointer to coordinate break parameters
            data = CastTo(row.SurfaceData, ZOSAPI.Editors.LDE.
                          ISurfaceCoordinateBreak.__name__)     # type: ZOSAPI.Editors.LDE.ISurfaceCoordinateBreak

            # Check that we have a coordinate system that is compatible with a coordinate break
            if cs.X.z != 0:
                raise ValueError('Nonzero z-translation')

            # Update the order
            if cs.translation_first:
                data.Order = 0
            else:
                data.Order = 1

            # Update the translation
            data.Decenter_X = cs.X.x
            data.Decenter_Y = cs.X.y

            # Update the rotation
            a, b, c = as_xyz_intrinsic_tait_bryan_angles(cs.Q)
            data.TiltAbout_X = a
            data.TiltAbout_Y = b
            data.TiltAbout_Z = c

        else:
            raise ValueError('Surface does not contain a coordinate break attribute')



    @property
    def tilt_dec(self) -> CoordinateSystem:
        return gcs()


