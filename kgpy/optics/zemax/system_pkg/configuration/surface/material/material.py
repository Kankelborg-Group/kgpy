from kgpy.optics.zemax import ZOSAPI

__all__ = ['Material', 'Mirror', 'EmptySpace']


class Material(kgpy.optics.system.surface.Material):

    def __init__(self, zemax_surface: ZOSAPI.Editors.LDE.ILDERow):

        super().__init__()

        zemax_surface.Material = ''


class Mirror(kgpy.optics.system.surface.material.Mirror, Material):

    def __init__(self, zemax_surface: ZOSAPI.Editors.LDE.ILDERow):

        super().__init__()
        Material.__init__(self, zemax_surface)

        zemax_surface.Material = 'MIRROR'


class EmptySpace(Material, kgpy.optics.system.surface.material.EmptySpace):

    pass
