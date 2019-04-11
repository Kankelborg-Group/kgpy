
from typing import List, Union, Dict
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

    def __init__(self, name: str, comment: str = ''):
        """
        Constructor for class kgpy.optics.Component
        :param name: Human-readable name of the component
        :param comment: Additional description of this component
        """

        # Save arguments as class variables
        self.name = name
        self.comment = comment

        # The surfaces within the component are stored as a linked list.
        # This attribute is a pointer to the first element of this list.
        self._surfaces = []          # type: List[Surface]
        
    @property
    def surfaces(self):
        return self._surfaces

    @property
    def first_surface(self) -> Union[Surface, None]:
        """
        :return: The first surface in the component if it exists, otherwise return None
        """

        # If the list of surfaces is not empty, return the first element
        if self._surfaces:
            return self._surfaces[0]

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
        return self._surfaces[-1].back_cs.X - self._surfaces[0].front_cs.X

    def append(self, surface: Surface) -> None:
        """
        Add provided surface to the end of the list of surfaces.
        :param surface: Surface to add to the end of the component
        :return: None
        """

        # Link the surface to this component instance
        surface.component = self

        # Append to the list of surfaces
        self._surfaces.append(surface)

    @property
    def _surfaces_dict(self) -> Dict[str, Surface]:
        """
        :return: A dictionary where the key is the surface name and the value is the surface.
        """

        # Allocate space for result
        d = {}

        # Loop through surfaces and add to dict
        for surf in self._surfaces:
            d[surf.name] = surf

        return d

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
    
    def __add__(self, other: 'Component'):
        
        c = Component(self.name + other.name)
        
        for surf in self._surfaces:
            
            c.append(surf)
            
        for surf in other.surfaces:
            
            c.append(surf)
            
        return c

    def __contains__(self, item: str) -> bool:
        """
        Check if this component contains a surface with the name item
        :param item: string to check
        :return: True if the component contains a surface with the name item.
        """

        return any(surf == item for surf in self._surfaces)

    def __getitem__(self, item: Union[int, str]) -> Surface:
        """
        Gets the surface at index i within the component, or the surface with the name item
        Accessed using the square bracket operator, e.g. surf = sys[i]
        :param item: Surface index or name of surface
        :return: Surface specified by item
        """

        # If the item is an integer, use it to access the surface list
        if isinstance(item, int):
            return self._surfaces.__getitem__(item)

        # If the item is a string, use it to access the surfaces dictionary.
        elif isinstance(item, str):
            return self._surfaces_dict.__getitem__(item)

        # Otherwise, the item is neither an int nor string and we throw an error.
        else:
            raise ValueError('Item is of an unrecognized type')

    def __delitem__(self, key: int):

        self[key].component = None

        self._surfaces.__delitem__(key)

    def __iter__(self):

        return self._surfaces.__iter__()

    @property
    def __len__(self):

        return self._surfaces.__len__

    def __str__(self) -> str:
        """
        :return: String representation of a component
        """

        # Construct line out of top-level parameters of the component
        ret = self.name + self.comment + ']\n'

        # Append lines for each surface within the component
        for surface in self._surfaces:
            ret = ret + '\t' + surface.__str__() + '\n'

        return ret

    __repr__ = __str__







