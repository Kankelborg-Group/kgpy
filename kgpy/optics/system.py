
import numpy as np
from typing import List
import quaternion as q

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

        # Initialize attributes to be set as surfaces are added.
        self.first_surface = None
        self.components = []        # type: List[Component]

    @property
    def surfaces(self) -> List[Surface]:
        """
        :return: An in-order list of all the surfaces in the system
        """

        # Initialize variables
        surf = self.first_surface
        surfaces = []

        # Follow links to next surface to construct list of surfaces, until link is None
        while surf is not None:

            # Append this surface to the list of surfaces to be returned
            surfaces.append(surf)

            # Select the next surface in the component
            surf = surf.next_surf_in_system

        return surfaces

    def append_component(self, component: Component) -> None:
        """
        Add the component and all its surfaces to the end of an optical system.
        :param component: component to be added to the system.
        :return: None
        """

        # Loop through the surfaces in the component and set the system links to be the same as the component links.
        for surf in component.surfaces:

            # Set two-way link
            surf.prev_surf_in_system = surf.prev_surf_in_component
            surf.next_surf_in_system = surf.next_surf_in_component

        # If the system already contains at least one surface
        if self.first_surface is not None:

            # Store pointers to the last surface currently in the system and to the first surface in the new component.
            last_surf = self.surfaces[-1]
            new_surf = component.surfaces[0]

            # Link the last surface in the system to the first surface in the component
            last_surf.next_surf_in_system = new_surf
            new_surf.prev_surf_in_system = last_surf

        # Otherwise the system contains no surfaces
        else:

            # Set the first surface in the system to the first surface in the component
            self.first_surface = component.first_surface

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

        for surf in self.surfaces:

            intercept = baffle_cs.xy_intercept(surf.front_cs.X, surf.back_cs.X)

            if intercept is not None:

                pass


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
            print(baffle_cs.zh)

            # Component of front face of surface perpendicular to baffle surface
            z0 = surface.front_cs.X.dot(baffle_cs.zh)
            print(surface.front_cs.X)
            print(z0)
            print(np.sign(z0))

            # Component of back face of surface perpendicular to baffle surface
            z1 = surface.back_cs.X.dot(baffle_cs.zh)
            print(surface.back_cs.X)
            print(z1)
            print(np.sign(z1))

            # If the two components above have different signs, the surface passes through the baffle surface
            if np.sign(z0) != np.sign(z1):

                # If there is an intersection, add it to the list of surfaces
                surfaces.append(surface)

        return surfaces

    def __str__(self) -> str:
        """
        :return: String representation of a system
        """

        # Construct line out of top-level parameters of the component
        ret = self.name + ', comment = ' + self.comment + '\n'

        # Append lines for each surface within the component
        for surface in self.surfaces:
            ret = ret + '\t' + surface.__str__() + '\n'

        return ret