import dataclasses
from kgpy.optics.system.surface import aperture
from .aperture import Aperture
from . import coordinate

__all__ = ['Decenterable']


class Decenterable(aperture.Decenterable, Aperture):

    @property
    def decenter(self) -> coordinate.Decenter:
        return self._decenter

    @decenter.setter
    def decenter(self, value: coordinate.Decenter):
        value.aperture = self
        self._decenter = value
