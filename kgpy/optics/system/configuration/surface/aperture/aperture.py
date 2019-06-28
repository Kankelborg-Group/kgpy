
import abc
import typing as tp
import astropy.units as u

from kgpy import optics

__all__ = ['Aperture']


class Aperture(abc.ABC):

    def __init__(self):

        self.surface = None

        self.is_obscuration = False

        self.decenter_x = 0 * u.mm      # type: u.Quantity
        self.decenter_y = 0 * u.mm      # type: u.Quantity

    @property
    def surface(self) -> tp.Optional[optics.system.configuration.Surface]:
        return self._surface

    @surface.setter
    def surface(self, value: tp.Optional[optics.system.configuration.Surface]):
        self._surface = value

    @property
    def is_obscuration(self) -> bool:
        return self._is_obscuration

    @is_obscuration.setter
    def is_obscuration(self, value: bool):
        self._is_obscuration = value

    @property
    def decenter_x(self) -> u.Quantity:
        return self._decenter_x

    @decenter_x.setter
    def decenter_x(self, value: u.Quantity):
        self._decenter_x = value

    @property
    def decenter_y(self) -> u.Quantity:
        return self._decenter_y

    @decenter_y.setter
    def decenter_y(self, value: u.Quantity):
        self._decenter_y = value
    
    @property
    @abc.abstractmethod
    def points(self) -> u.Quantity:
        pass













