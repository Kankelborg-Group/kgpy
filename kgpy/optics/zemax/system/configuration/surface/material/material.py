
from kgpy import optics

__all__ = ['Material', 'Mirror', 'EmptySpace']


class Material(optics.system.configuration.surface.Material):

    def __init__(self):

        super().__init__()

        self.zos_str = ''


class Mirror(Material, optics.system.configuration.surface.material.Mirror):

    def __init__(self):

        super().__init__()

        self.zos_str = 'MIRROR'


class EmptySpace(Material, optics.system.configuration.surface.material.EmptySpace):

    pass
