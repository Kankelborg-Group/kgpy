
__all__ = ['Component']


class Component:
    """
    An optical component is a collection of one or more surfaces such as the tips/tilts, clear aperture, and the optical
    surface of an optical element.
    """

    def __init__(self, name, surfaces):
        """
        Constructor for class kgpy.optics.Component
        :param name: Human-readable name of the component
        :param surfaces: List of initial surfaces within the component
        :type name: str
        :type surfaces: list[kgpy.optics.Surface]
        """

        self.name = name
        self.surfaces = surfaces






