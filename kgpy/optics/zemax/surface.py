
from typing import List
import astropy.units as u

from kgpy.optics import Surface
from kgpy.optics.zemax.ZOSAPI.Editors.LDE import ILDERow

__all__ = ['ZmxSurface']


class ZmxSurface(Surface):
    """
    Child of the Surface class which acts as a wrapper around the Zemax API ILDERow class (a Zemax surface).
    The ZmxSurface class is intended to be used temporarily as the functionality in the ILDERow class is implemented in
    the Surface superclass.
    """

    def __init__(self, zmx_surf: List[ILDERow], length_units: u.Unit):
        """
        Constructor for ZmxSurface object.
        :param zmx_surf: Pointer to the zmx_surf to wrap this class around
        """

        # Save arguments to class variables
        self._zmx_surf = zmx_surf
        self._u = length_units

    @property
    def name(self) -> str:
        """
        Grab the name section of the comment string
        :return: Human-readable name of the surface
        """

    @property
    def comment(self):
        return self._zmx_surf.Comment

    @property
    def _comment_str(self) -> str:
        """
        Grab the entire comment string from Zemax, so we can split it up into self.name and self.comment.
        This lets us be consistent with the surface interface and also express the concept of a surface name and comment
        in Zemax.
        In Zemax the syntax is <name>:<comment>
        :return: Zemax comment string
        """
        return self._zmx_surf.Comment

    @property
    def system_index(self):
        """
        :return: The index of this surface within the overall optical system
        """
        return self._zmx_surf.RowIndex

    @property
    def thickness(self):
        return self._zmx_surf.Thickness

    @thickness.setter
    def thickness(self, t):
        self._zmx_surf.Thickness = t
