
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
    def last_surface(self):



    @property
    def T(self) -> Vector:
        """
        Thickness vector
        :return: Vector pointing from center of a component's front face to the center of a component's back face.
        """

        return self.back_cs.X - self.cs.X

    def append_surface(self, surface: Surface) -> int:
        """
        Add provided surface to the specified list of surfaces.
        :param surface: Surface to add to the end of the component
        :return:
        """

        # If there is already at least one surface in the component, populate the previous_surf attribute in this
        # surface
        if self.surfaces:
            surface.previous_surf = self.surfaces[-1]

        # Add a pointer to this component to the surface
        surface.component = self

        # Append updated surface to list of surfaces
        self.surfaces.append(surface)

    def __str__(self):
        """
        :return: String representation of a component
        """

        # Construct line out of top-level parameters of the component
        ret = self.name + ', comment = ' + self.comment + ', cs = [' + self.cs.__str__() + ']\n'

        # Append lines for each surface within the component
        for surface in self.surfaces:
            ret = ret + '\t' + surface.__str__() + '\n'

        return ret







