
from typing import List
from . import Surface


__all__ = ['Component']


class Component(Surface):
    """
    An optical component is a collection of one or more surfaces such as the tips/tilts, clear aperture, and the optical
    surface of an optical element.
    Note that the surfaces within the component do not have to be in order
    """

    def __init__(self, name: str, surfaces: List[Surface], comment: str = ''):
        """
        Constructor for class kgpy.optics.Component
        :param name: Human-readable name of the component
        :param surfaces: List of initial surfaces within the component
        :param comment: Additional description of this component
        """

        super().__init__(name, comment=comment)
        self.surfaces = []

        for surface in surfaces:
            self.append_surface(surface)

    # @property
    # def thickness(self):
    #     """
    #     Total thickness of the component
    #     :return: Sum of every surface's thickness
    #     :rtype: float
    #     """
    #     t = 0
    #     for surface in self.surfaces:
    #         t += surface.thickness

    def append_surface(self, surface: Surface) -> int:
        """
        Add provided surface to the specified list of surfaces.
        Currently, the main reason for this method is to ensure that the global coordinate of each surface is set
        correctly.
        :param surface:
        :return:
        """

        # Append updated surface to list of surfaces
        self.surfaces.append(surface)









