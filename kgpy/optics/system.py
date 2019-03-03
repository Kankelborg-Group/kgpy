
import numpy as np
from typing import List, Dict, Union
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

        # Initialize the object surface.
        # This surface is not usually considered in the list of surfaces, since it often has infinite thickness.
        # Therefore there is a separate pointer for this surface, instead of it being the self.first_surface.
        self.obj_surface = Surface('Object', thickness=np.inf * u.mm)
        self.obj_surface.is_object = True

        # Initialize attributes to be set as surfaces are added.
        self.surfaces = []     # type: List[Surface]

    @property
    def first_surface(self) -> Union[Surface, None]:
        """
        :return: The first optical surface in the system if it exists, otherwise return None
        """

        # If there is at least one surface in the list, return the first element of the list
        if self.surfaces:
            return self.surfaces[0]

        # Otherwise the list is empty, and we return None
        else:
            return None

    @property
    def components(self) -> Dict[str, Component]:
        """
        :return: A Dictionary with all the Components in the system as values and their names as the keys.
        """

        # Allocate space to store the new dictionary
        comp = {}

        # Loop through all the surfaces in the system
        for surf in self.surfaces.values():

            # Add this surface's component to the dictionary if it's not already there.
            if surf.component.name not in comp:
                comp[surf.component.name] = surf.component

        return comp

    def insert_surface(self, surface: Surface, index: int) -> None:
        """
        Insert a surface into the specified position index the system
        :param surface: Surface object to be added to the system
        :param index: Index that we want the object to be placed at
        :return: None
        """

        surface.sys = self
        self.surfaces.insert(index, surface)

    def append_surface(self, surface: Surface) -> None:
        """
        Add a surface to the end of an optical system
        :param surface: The surface to be added
        :return: None
        """

        surface.sys = self
        self.surfaces.append(surface)

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
        This function assumes that the baffle lies in the x-y plane of this coordinate system.
        :return: Pointer to Baffle component
        """

        # Create new component to store the baffle
        baffle = Component(baffle_name)

        # Define variable to track how many times the system intersected the
        baffle_pass = 0

        # Make a copy of the surfaces list so we don't try to iterate over and write to the same list
        old_surfaces = self.surfaces.copy()

        # Loop through all surfaces in the system to see if any intersect with a baffle
        for surf in old_surfaces:

            # Compute the intersection between the thickness vector and the x-y plane of the baffle, if it exists.
            intercept = baffle_cs.xy_intercept(surf.front_cs.X, surf.back_cs.X)

            # If the intercept exists, insert the new baffle
            if intercept is not None:

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
                self.insert_surface(baffle_surf, surf.system_index + 1)

                # Insert new baffle surface into baffle component
                baffle.append_surface(baffle_surf)

                # Update the number of baffle passes
                baffle_pass += 1

        return baffle

    def to_zemax(self):

        pass

    def __str__(self) -> str:
        """
        :return: String representation of a system
        """

        # Construct line out of top-level parameters of the component
        ret = self.name + ', comment = ' + self.comment + '\n'

        ret += '\t' + self.obj_surface.__str__() + '\n'

        # Append lines for each surface within the component
        for surface in self.surfaces:
            ret = ret + '\t' + surface.__str__() + '\n'

        return ret