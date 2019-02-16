
import numpy as np
from typing import List
import quaternion as q

from kgpy.math import CoordinateSystem
from kgpy.math.coordinate_system import GlobalCoordinateSystem as gcs
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

    def add_baffle(self, baffle_name: str, baffle_cs: CoordinateSystem) -> Component:
        """
        Add a baffle to the system at the specified coordinate system across the x-y plane.
        This function automatically calculates how many times the raypath crosses the baffle plane, and constructs the
        appropriate amount of baffle surfaces
        :param baffle_cs: Coordinate system of the baffle
        :return: Pointer to Baffle component
        """

        # Create new component to store the baffle
        baffle = Component(baffle_name)

        # Define variable to track how many times the system intersected the
        baffle_pass = 0

        # Loop through all surfaces in the system to see if any intersect with a baffle
        for surf in self.surfaces:

            # Compute the intersection between the thickness vector and the x-y plane of the baffle, if it exists.
            intercept = baffle_cs.xy_intercept(surf.front_cs.X, surf.back_cs.X)

            # If the intercept exists, insert the new baffle
            if intercept is not None:

                # Grab pointer to the surface after the current surface in the system
                next_surf = surf.next_surf_in_system

                # Compute the new thickness vectors for both to
                t1 = intercept - surf.front_cs.X  # New thickness of original surface
                t2 = surf.back_cs.X - intercept   # Thickness of new surface to be added after the baffle

                # Modify the original surface to have the correct thickness
                surf.thickness = t1.dot(surf.front_cs.zh)

                # Calculate the tilt/decenter required to put the baffle in the correct place
                cs = baffle_cs.diff(surf.back_cs)

                # Create new baffle surface
                baffle_thickness = t2.dot(surf.front_cs.zh)
                baffle_surf = Surface(baffle_name, comment='pass = ' + str(baffle_pass), thickness=baffle_thickness,
                                      tilt_dec=cs)

                # Link the new baffle surface into the system
                surf.next_surf_in_system = baffle_surf
                baffle_surf.next_surf_in_system = next_surf
                baffle_surf.prev_surf_in_system = surf
                if next_surf is not None:
                    next_surf.prev_surf_in_system = baffle_surf

                # Insert new baffle surface into baffle component
                baffle.append_surface(baffle_surf)

                # Update the number of baffle passes
                baffle_pass += 1

        # Insert new baffle component into system
        self.components.append(baffle)

        return baffle

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