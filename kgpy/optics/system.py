
import numpy as np
from typing import List
import quaternion as q
import astropy.units as u

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
        self.first_surface = None   # type: Surface
        self.components = []        # type: List[Component]

        # Initialize the object as the first surface in the system
        # obj_comp = Component('Object', matching_surf=True)
        # obj_comp.first_surface.thickness = np.inf * u.mm        # The object defaults to being at infinity
        # self.append_component(obj_comp)

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

    def append_surface(self, surface: Surface) -> None:

        # If there is at least one surface in the system, append the new surface after the last surface
        if self.surfaces:

            # Store pointer to the current last surface in the system, so we can link the current system to it.
            last_surf = self.surfaces[-1]

            # Set the two-way link between the current last surface and the new surface
            last_surf.next_surf_in_system = surface
            surface.prev_surf_in_system = last_surf

        # Otherwise, the system is empty and this surface is the first surface.
        else:
            self.first_surface = surface

    def append_component(self, component: Component) -> None:
        """
        Add the component and all its surfaces to the end of an optical system.
        :param component: component to be added to the system.
        :return: None
        """

        # Link the system to the component
        component.sys = self

        # Loop through the surfaces in the component add them to the back of the system
        for surf in component.surfaces:
            self.append_surface(surf)

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
        :param baffle_name: Human-readable name of the baffle
        :param baffle_cs: Coordinate system where the baffle will be placed.
        This function assumes that the baffle lies the x-y plane of this coordinate system.
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

    def to_zemax(self):

        pass

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