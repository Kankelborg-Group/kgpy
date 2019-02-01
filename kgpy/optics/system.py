
from abc import ABC, abstractmethod
from unittest import TestCase
import numpy as np
from typing import List
from copy import deepcopy

from kgpy.math import CoordinateSystem
from kgpy.math.coordinate_system import GlobalCoordinateSystem
from kgpy.optics import Baffle, Surface, Component

__all__ = ['System']


class System:
    """
    The System class simulates an entire optical system, and is represented as a series of Components.
    This class is intended to be a drop-in replacement for a Zemax system.
    """

    def __init__(self, name: str, components: List[Component]):
        """
        Define an optical system by providing a name and a list of Components.
        The surfaces within each Component are added sequentially.
        :param name: Human-readable name of the system
        :param components: List of initial components for the system
        """

        # Save input arguments to class variables
        self.name = name
        self.components = components

        # Define the coordinate system of the System as the origin of the global coordinate system
        self.cs = GlobalCoordinateSystem()

        # Create surfaces variable to store list
        self.surfaces = []      # type: List[Surface]

    def append_surface(self, surface: Surface) -> None:
        """
        Add the surface to the end of the surface list
        :param surface: Surface to be appended
        :return: None
        """

        self.surfaces.append(surface)

    def append_component(self, component: Component) -> None:
        """
        Add the component to the end of the component list.
        This function implicitly adds each surface inside the component to the surfaces list in order.
        :param component: component to be added to the component list
        :return: None
        """

        # If the list of components is empty, the coordinate system of the component is the same as the coordinate
        # system of the System.
        # Otherwise the coordinate system of the component is the the coordinate system of the last component translated
        # by the thickness vector of the component
        if not self.components:
            component.cs = deepcopy(self.cs)
        else:
            last_comp = self.surfaces[-1]
            surface.cs = last_surf.cs + last_surf.T

        # Loop through every surface in the component and add it to the full list of surfaces
        for surface in component.surfaces:

            self.append_surface(surface)

        # Append this component to the list of components
        self.components.append(component)

            




    # @property
    # def surfaces(self):
    #     """
    #     List of all surfaces in the system, in order.
    #     This is done by concatenating all surfaces from each component together.
    #     :return: List of all surfaces in the system
    #     :rtype: list[kgpy.optics.Surface]
    #     """
    #     s = []
    #     for component in self.components:
    #         for surface in component.surfaces:
    #             s.append(surface)
    #
    #     return s

    def add_baffle(self, baffle_cs: CoordinateSystem) -> Baffle:
        """
        Add a baffle to the system at the specified coordinate system across the x-y plane.
        This function automatically calculates how many times the raypath crosses the baffle plane, and constructs the
        appropriate amount of baffle surfaces
        :param baffle_cs: Coordinate system of the baffle
        :return: Pointer to Baffle component
        """

        # Calculate the list of surfaces that we need to split
        surfaces = self.calc_surface_intersections(baffle_cs)

        # Loop through each surface, shorten it and add the baffle to fill the remaining space
        for surface in surfaces:

            pass




    @abstractmethod
    def calc_surface_intersections(self, baffle_cs: CoordinateSystem) -> List[Surface]:
        """
        This function is used to determine which surfaces to split when inserting a baffle.
        The optics model is sequential, so if the baffle is used by rays going in different directions, we need to model
        the baffle as multiple surfaces.
        :return: List of surface indices which intersect a baffle
        """

        # Initialize looping variables
        surfaces = []  # List of surface indices which cross a baffle

        # Loop through every surface and keep track of how often we cross the global z coordinate of the baffle
        for surface in self.surfaces:

            # Component of front face of surface perpendicular to baffle surface
            z0 = np.dot(surface.cs.X, baffle_cs.zh)

            # Component of back face of surface perpendicular to baffle surface
            z1 = np.dot(surface.T, baffle_cs.zh)

            # If the two components above have different signs, the surface passes through the baffle surface
            if np.sign(z0) != np.sign(z1):

                # If there is an intersection, add it to the list of surfaces
                surfaces.append(surface)

        return surfaces
