
import abc
import typing as tp
import astropy.units as u

from kgpy import math, optics

__all__ = ['Aperture']


class Aperture(abc.ABC):

    @property
    @abc.abstractmethod
    def points(self) -> u.Quantity:
        pass













