
from typing import List
from copy import deepcopy
import astropy.units as u

from . import Surface
from kgpy.math import Vector
from kgpy.math.coordinate_system import GlobalCoordinateSystem

__all__ = ['Component']


class Component:
    """
    An optical component is a collection of one or more surfaces such as the tips/tilts, clear aperture, and the optical
    surface of an optical element.
    Note that the surfaces within the component do not have to be in order
    """

    def __init__(self, name: str, comment: str = ''):
        """
        Constructor for class kgpy.optics.Component
        :param name: Human-readable name of the component
        :param comment: Additional description of this component
        """

        # Save arguments as class variables
        self.name = name
        self.comment = comment

        # Initialize the list of surfaces to an empty list
        self.surfaces = []      # type: List[Surface]

        # Initialize the coordinate system to the global coordinate system
        self.cs = GlobalCoordinateSystem()


    @property
    def T(self):
        """
        Thickness vector
        :return: Vector pointing from center of a component's front face to the center of a component's back face.
        """

        # If the component contains at least one surface, return the difference between the vector pointing the back
        # face of the component and the vector pointing to the front face of the component.
        # Else, the component contains no surfaces and the thickness is the zero vector.
        if self.surfaces:
            s0 = self.surfaces[0]
            s1 = self.surfaces[-1]
            return (s1.cs.X + s1.T) - s0.cs.X
        else:
            return Vector([0, 0, 0] * u.mm)

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
            surface.cs = deepcopy(self.cs)
        else:
            last_surf = self.surfaces[-1]
            surface.cs = last_surf.cs + last_surf.T

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







