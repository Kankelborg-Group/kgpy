import abc
import dataclasses
import numpy as np
import astropy.units as u
from kgpy import mixin, vector, optimization
from .. import Rays

__all__ = ['Sag']


@dataclasses.dataclass
class Sag(
    mixin.Broadcastable,
    mixin.Copyable,
    abc.ABC,
):

    @abc.abstractmethod
    def __call__(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        pass

    @abc.abstractmethod
    def normal(self, x: u.Quantity, y: u.Quantity) -> u.Quantity:
        pass
