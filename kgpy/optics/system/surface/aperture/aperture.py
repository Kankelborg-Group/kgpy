
import abc
import typing as tp
import astropy.units as u

from kgpy import math, optics

__all__ = ['Aperture']


class Aperture(abc.ABC):

    def __init__(self, is_obscuration: bool = False,):

        self._is_obscuration = is_obscuration

    @property
    def is_obscuration(self) -> bool:
        return self._is_obscuration

    @property
    @abc.abstractmethod
    def points(self) -> u.Quantity:
        pass













