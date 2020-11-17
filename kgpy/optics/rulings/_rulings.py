import typing as typ
import abc
import astropy.units as u
from kgpy import mixin, vector
from .. import Rays
from ..material import Material

__all__ = ['Rulings']


class Rulings(
    mixin.Broadcastable,
    mixin.Copyable,
    abc.ABC,
):

    @abc.abstractmethod
    def normal(self, x: u.Quantity, y: u.Quantity):
        pass

    @abc.abstractmethod
    def _effective_input_vector(
            self,
            rays: Rays,
            material: typ.Optional[Material] = None,
    ) -> u.Quantity:
        pass

    def effective_input_direction(
            self,
            rays: Rays,
            material: typ.Optional[Material] = None,
    ):
        return vector.normalize(self._effective_input_vector(rays, material=material))

    def effective_input_index(
            self,
            rays: Rays,
            material: typ.Optional[Material] = None,
    ):
        return vector.length(self._effective_input_vector(rays, material=material))
