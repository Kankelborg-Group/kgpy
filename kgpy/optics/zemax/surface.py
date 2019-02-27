
from typing import List, Dict
import astropy.units as u

from kgpy.optics import Surface
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.ZOSAPI.Editors.LDE import ILDERow

__all__ = ['ZmxSurface']


class ZmxSurface(Surface):
    """
    Child of the Surface class which acts as a wrapper around the Zemax API ILDERow class (a Zemax surface).
    The ZmxSurface class is intended to be used temporarily as the functionality in the ILDERow class is implemented in
    the Surface superclass.
    """

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
        self.component = None                  # type: kgpy.optics.Component

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
    def cs_break(self):

        row = self._get_attr_row('cs_break')

        if row.Type == ZOSAPI.Editors.LDE.SurfaceType.CoordinateBreak:
            row.


