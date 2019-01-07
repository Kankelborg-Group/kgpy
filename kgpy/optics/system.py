
from abc import ABC, abstractmethod
from unittest import TestCase


__all__ = ['System']


class System:
    """
    The System class simulates an entire optical system, and is represented as a series of Components.
    This class is intended to be a drop-in replacement for a Zemax system.
    """

    def __init__(self, name, components):
        """
        Define an optical system by providing a name and a list of Components.
        The surfaces within each Component are added sequentially.
        :param name: Human-readable name of the system
        :param components: List of initial components for the system
        :type name: str
        :type components: list[kgpy.optics.Component]
        """

        # Save input arguments to class variables
        self.name = name
        self.components = components

    @property
    def surfaces(self):
        """
        List of all surfaces in the system, in order.
        This is done by concatenating all surfaces from each component together.
        :return: List of all surfaces in the system
        :rtype: list[kgpy.optics.Surface]
        """
        s = []
        for component in self.components:
            for surface in component.surfaces:
                s.append(surface)

        return s


    def add_baffle(self, global_z):
        """
        Add a baffle to the system at the specified global z-coordinate across the x-y plane.
        This function automatically calculates how many times the raypath crosses the baffle plane, and constructs the
        appropriate amount of baffle surfaces
        :param global_z: Global z-coordinate of the baffle plane.
        :type global_z: float
        :return:
        """

        # Calculate the list of surfaces that we need to split
        surfaces = self.calc_surface_intersections(global_z)

        # Loop through each surface, shorten it and add the baffle to fill the remaining space
        for surface in surfaces:

            surface




    @abstractmethod
    def calc_surface_intersections(self, global_z):
        """
        This function is used to determine which surfaces to split when inserting a baffle.
        The optics model is sequential, so if the baffle is used by rays going in different directions, we need to model
        the baffle as multiple surfaces.
        :return: List of surface indices which intersect a baffle
        :rtype: list[int]
        """

        # Initialize looping variables
        z = 0  # Test z-coordinate
        z_is_greater = False  # Flag to store what side of the baffle the test coordinate was on in the last iteration
        surfaces = []  # List of surface indices which cross a baffle

        # Loop through every surface and keep track of how often we cross the global z coordinate of the baffle
        for surface in self.surfaces:

            # Update test z-coordinate
            z += surface.thickness

            # Check if the updated test coordinate has crossed the baffle coordinate since the last iteration.
            # If so, append surface to list of intersecting surfaces
            if z_is_greater:  # Crossing from larger to smaller
                if z < global_z:
                    surfaces.append(surface)
                    z_is_greater = False

            else:  # Crossing from smaller to larger
                if z > global_z:
                    surfaces.append(surface)
                    z_is_greater = True

        return surfaces


