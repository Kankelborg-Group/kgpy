
from typing import List
from copy import deepcopy
import astropy.units as u

from . import Surface
from kgpy.math import Vector, CoordinateSystem
from kgpy.math.coordinate_system import GlobalCoordinateSystem as gcs

__all__ = ['Component']


class Component:
    """
    An optical component is a collection of one or more surfaces such as the tips/tilts, clear aperture, and the optical
    surface of an optical element.
    A Component is part of a System, the hierarchy is as follows: System -> Component -> Surface.
    Note that the surfaces within the component do not have to be contiguous.
    Note: If the surfaces within a component are not contiguous, it is not clear what is meant by rotating or
    translating the Component.
    """

    def __init__(self, name: str, comment: str = '', cs_break: CoordinateSystem = gcs()):
        """
        Constructor for class kgpy.optics.Component
        :param name: Human-readable name of the component
        :param comment: Additional description of this component
        :param cs_break: Coordinate system that is applied to the component, and permanently modifies the current
        Coordinate system.
        This feature can be used to rotate and translate the whole component, with all the surfaces within.
        """

        # Save arguments as class variables
        self.name = name
        self.comment = comment
        self.cs_break = cs_break

        # The surfaces within the component are stored as a linked list.
        # This attribute is a pointer to the first element of this list
        self.first_surface = None   # type: Surface

    @property
    def surfaces(self) -> List[Surface]:
        """
        :return: An in-order list of all the surfaces in the component
        """

        # Initialize variables
        surf = self.first_surface
        surfaces = []

        # Follow links to next surface to construct list of surfaces, until link is None
        while surf is not None:

            # Append this surface to the list of surfaces to be returned
            surfaces.append(surf)

            # Select the next surface in the component
            surf = surf.next_surf_in_component

        return surfaces

    @property
    def T(self) -> Vector:
        """
        Thickness vector
        :return: Vector pointing from center of a component's front face to the center of a component's back face.
        """

        # Subtract translation vector from the back face of the last surface in the component from the translation
        # vector to the first surface of the component.
        return self.surfaces[-1].back_cs.X - self.surfaces[0].front_cs.X

    def append_surface(self, s: Surface):
        """
        Add provided surface to the end of the list of surfaces.
        :param s: Surface to add to the end of the component
        :return: None
        """

        # Link the surface to this component instance
        s.component = self

        # If the list of surfaces is not empty, add this surface to the end.
        if self.surfaces:

            # Grab pointer to the last surface in the component
            last_surf = self.surfaces[-1]

            # Link up last surface and the new surface appropriately
            s.prev_surf_in_component = last_surf
            last_surf.next_surf_in_component = s

        # Otherwise s is the first surface in the component
        else:
            self.first_surface = s

    def __str__(self) -> str:
        """
        :return: String representation of a component
        """

        # Construct line out of top-level parameters of the component
        ret = self.name + ', comment = ' + self.comment + ']\n'

        # Append lines for each surface within the component
        for surface in self.surfaces:
            ret = ret + '\t' + surface.__str__() + '\n'

        return ret







