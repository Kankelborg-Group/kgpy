
from kgpy import optics
from kgpy.optics.zemax import ZOSAPI

__all__ = ['Material', 'Mirror', 'EmptySpace']


class Material(optics.system.configuration.surface.Material):

    def __init__(self, zemax_surface: ZOSAPI.Editors.LDE.ILDERow):

        super().__init__()

        zemax_surface.Material = ''


class Mirror(optics.system.configuration.surface.material.Mirror, Material):

    def __init__(self, zemax_surface: ZOSAPI.Editors.LDE.ILDERow):

        super().__init__()
        Material.__init__(self, zemax_surface)

        zemax_surface.Material = 'MIRROR'


class EmptySpace(Material, optics.system.configuration.surface.material.EmptySpace):

    pass
