
from typing import List, Dict
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

    def __init__(self, name: str, zmx_surf: ILDERow, length_units: u.Unit):
        """
        Constructor for ZmxSurface object.
        :param zmx_surf: Pointer to the zmx_surf to wrap this class around
        """

        # Call the superclass to initialize most of the class variables.
        super().__init__(name)

        # Save remaining arguments to class variables
        self.zmx_surf = zmx_surf
        self.u = length_units

        # Initialize class variables
        self.attr_surfaces = {}        # type: Dict[str, ILDERow]

    @property
    def thickness(self):
        return self.zmx_surf.Thickness
