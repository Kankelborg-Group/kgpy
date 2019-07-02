
from kgpy import optics
from kgpy.optics.zemax import ZOSAPI

__all__ = ['Material', 'Mirror', 'EmptySpace']


class Material(optics.system.configuration.surface.Material):

    def __init__(self, zemax_surface: ZOSAPI.Editors.LDE.ILDERow):

        super().__init__()

        self._zemax_surface = zemax_surface

        self._zos_str = ''

    @property
    def zemax_surface(self) -> ZOSAPI.Editors.LDE.ILDERow:
        return self._zemax_surface

    @property
    def zemax_str(self) -> str:
        return self._zos_str


class Mirror(optics.system.configuration.surface.material.Mirror, Material):

    def __init__(self, zemax_surface: ZOSAPI.Editors.LDE.ILDERow):

        super().__init__()
        Material.__init__(self, zemax_surface)

        self._zos_str = 'MIRROR'


class EmptySpace(Material, optics.system.configuration.surface.material.EmptySpace):

    pass
