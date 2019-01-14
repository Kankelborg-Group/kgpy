
from . import Surface

__all__ = ['Component']


class Component(Surface):
    """
    An optical component is a collection of one or more surfaces such as the tips/tilts, clear aperture, and the optical
    surface of an optical element.
    """

    def __init__(self, name, surfaces, comment=''):
        """
        Constructor for class kgpy.optics.Component
        :param name: Human-readable name of the component
        :param surfaces: List of initial surfaces within the component
        :param comment: Additional description of this component
        :type name: str
        :type surfaces: list[kgpy.optics.Surface]
        :type comment: str
        """

        super().__init__(name, comment=comment)
        self.surfaces = []

        for surface in surfaces:
            self.append_surface(surface)

    @property
    def thickness(self):
        """
        Total thickness of the component
        :return: Sum of every surface's thickness
        :rtype: float
        """
        t = 0
        for surface in self.surfaces:
            t += surface.thickness

    def append_surface(self, surface):
        """
        Add provided surface to the specified list of surfaces.
        Currently, the main reason for this method is to ensure that the global coordinate of each surface is set
        correctly.
        :param surface:
        :return:
        """

        # Set the global z coordinate of the surface to z value within the component plus the total z offset of the
        # component
        surface.z = self.thickness + self.z

        # Append updated surface to list of surfaces
        self.surfaces.append(surface)









