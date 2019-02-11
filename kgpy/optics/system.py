
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

    def __init__(self, name: str, comment: str = ''):
        """
        Define an optical system by providing a name.
        :param name: Human-readable name of the system
        :param comment: Additional information about the system.
        """

        # Save input arguments to class variables
        self.name = name
        self.comment = comment

        # Initialize empty list of components
        self.components = []        # type: List[Component]

    def append_component(self, component: Component) -> None:
        """
        Add the component to the end of the component list.
        :param component: component to be added to the component list
        :return: None
        """

        # If the system contains at least one component, set the populate the previous surface attribute of the
        # component we are adding to the list.
        if self.components:
            component.previous_component = self.components[-1]

        # Append this component to the list of components
        self.components.append(component)

            
    def zipper_component(self, component: Component, indices: List[int]) -> None:
        """
        Places the surfaces within a component in the system locations specified by indices.
        This function allows for the use of non-sequential components.
        :param component: Component to zipper into the system
        :param indices: Index for each surface in the component describing where in the system that surface should be
        inserted
        :return: None
        """

        pass


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
