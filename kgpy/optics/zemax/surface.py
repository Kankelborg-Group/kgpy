
from unittest import TestCase

from kgpy.optics import Surface

__all__ = ['ZmxSurface']


class ZmxSurface(Surface):
    """
    Child of the Surface class which acts as a wrapper around the Zemax API ILDERow class (a Zemax surface).
    The ZmxSurface class is intended to be used temporarily as the functionality in the ILDERow class is implemented in
    the Surface superclass.
    """

    def __init__(self, name, zmx_surf):
        """
        Constructor for ZmxSurface object. Currently only saves arguments to class variables
        :param zmx_surf: Pointer to the zmx_surf to wrap this class around
        :type zmx_surf:
        """

        self.name = name
        self.zmx_surf = zmx_surf

    @property
    def comment(self):
        return self.zmx_surf.Comment

    @property
    def thickness(self):
        return self.zmx_surf.Thickness

    @thickness.setter
    def thickness(self, t):
        self.zmx_surf.Thickness = t
