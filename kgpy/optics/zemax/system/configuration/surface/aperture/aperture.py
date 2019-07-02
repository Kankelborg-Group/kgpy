
import abc
import typing as tp
import astropy.units as u

from kgpy import optics
from kgpy.optics.zemax import ZOSAPI


__all__ = ['Aperture']


class Aperture(optics.system.configuration.surface.Aperture):

    def __init__(self, zemax_surface: ZOSAPI.Editors.LDE.ILDERow):

        super().__init__()

        self._zemax_surface = zemax_surface

    @property
    def zemax_surface(self) -> ZOSAPI.Editors.LDE.ILDERow:
        return self._zemax_surface














