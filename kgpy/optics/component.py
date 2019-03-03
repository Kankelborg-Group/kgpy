
from typing import List, Union
from copy import deepcopy
import astropy.units as u

import kgpy.optics
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

    def __init__(self, name: str, comment: str = '', cs_break: CoordinateSystem = gcs(), matching_surf=False):
        """
        Constructor for class kgpy.optics.Component
        :param name: Human-readable name of the component
        :param comment: Additional description of this component
        :param cs_break: Coordinate system that is applied to the component, and permanently modifies the current
        Coordinate system.
        This feature can be used to rotate and translate the whole component, with all the surfaces within.
        :param matching_surf: Flag which allows for the component to populated with single surface with the same name as
        the component.
        This feature is a shorthand for creating one-surface components.
        """

        # Save arguments as class variables
        self.name = name
        self.comment = comment
        self.cs_break = cs_break

        # The surfaces within the component are stored as a linked list.
        # This attribute is a pointer to the first element of this list.
        self.surfaces = []          # type: List[Surface]

        # If the matching surface flag is set, create the matching surface and add it to the component.
        if matching_surf:
            s = Surface(name)
            self.append_surface(s)

    @property
    def first_surface(self) -> Union[Surface, None]:
        """
        :return: The first surface in the component if it exists, otherwise return None
        """

        # If the list of surfaces is not empty, return the first element
        if self.surfaces:
            return self.surfaces[0]

        # Otherwise return None
        else:
            return None

    @property
    def T(self) -> Vector:
        """
        Thickness vector
        :return: Vector pointing from center of a component's front face to the center of a component's back face.
        """

        # Subtract translation vector from the back face of the last surface in the component from the translation
        # vector to the first surface of the component.
        return self.surfaces[-1].back_cs.X - self.surfaces[0].front_cs.X

    def append_surface(self, surface: Surface) -> None:
        """
        Add provided surface to the end of the list of surfaces.
        :param surface: Surface to add to the end of the component
        :return: None
        """

        # Link the surface to this component instance
        surface.component = self

        # Append to the list of surfaces
        self.surfaces.append(surface)

    def __eq__(self, other: 'Component') -> bool:
        """
        Check if two components are equal
        :param other: The other component to compare to this one.
        :return: True if the two components are equal, False otherwise.
        """

        if other is not None:

            # Compare component name and comment for now.
            a = self.name == other.name
            b = self.comment == other.comment

            return a and b

        else:

            return False

    def __contains__(self, item: str) -> bool:
        """
        Check if this component contains a surface with the name item
        :param item: string to check
        :return: True if the component contains a surface with the name item.
        """

        return any(surf == item for surf in self.surfaces)

    def __str__(self) -> str:
        """
        :return: String representation of a component
        """

        # Construct line out of top-level parameters of the component
        ret = self.name + self.comment + ']\n'

        # Append lines for each surface within the component
        for surface in self.surfaces:
            ret = ret + '\t' + surface.__str__() + '\n'

        return ret

    __repr__ = __str__







