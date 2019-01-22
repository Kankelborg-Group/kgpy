
from typing import List
from copy import deepcopy
from . import Surface

from kgpy.math import GlobalCoordinateSystem

__all__ = ['Component']


class Component:
    """
    An optical component is a collection of one or more surfaces such as the tips/tilts, clear aperture, and the optical
    surface of an optical element.
    Note that the surfaces within the component do not have to be in order
    """

    def __init__(self, name: str, surfaces: List[Surface], comment: str = ''):
        """
        Constructor for class kgpy.optics.Component
        :param name: Human-readable name of the component
        :param surfaces: List of initial surfaces within the component
        :param comment: Additional description of this component
        """

        # Save arguments as class variables
        self.name = name
        self.comment = comment

        # Initialize the list of surfaces to an empty list
        self.surfaces = []

        # Initialize the coordinate system to the global coordinate system
        self.cs = GlobalCoordinateSystem()

        # Loop through every surface and add it to the list of surfaces within this component
        for surface in surfaces:
            self.append_surface(surface)

    # @property
    # def thickness(self):
    #     """
    #     Total thickness of the component
    #     :return: Sum of every surface's thickness
    #     :rtype: float
    #     """
    #     t = 0
    #     for surface in self.surfaces:
    #         t += surface.thickness

    def append_surface(self, surface: Surface) -> int:
        """
        Add provided surface to the specified list of surfaces.
        Currently, the main reason for this method is to ensure that the global coordinate of each surface is set
        correctly.
        :param surface:
        :return:
        """

        # If the list of surfaces is empty, the coordinate system of the surface is the same as the coordinate system of
        # the component.
        # Otherwise the coordinate system of the surface is the coordinate system of the last surface in the list,
        # translated by the thickness vector
        if not self.surfaces:
            surf_cs = deepcopy(self.cs)
            surf_cs =
            surface.cs =

        # Append updated surface to list of surfaces
        self.surfaces.append(surface)









