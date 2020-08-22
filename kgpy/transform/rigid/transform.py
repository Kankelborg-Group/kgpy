import typing as typ
import abc
import numpy as np
import astropy.units as u
from kgpy import vector
from .. import Transform as GeneralTransform

__all__ = ['Transform']


class Transform(GeneralTransform):

    @property
    @abc.abstractmethod
    def rotation_eff(self) -> u.Quantity:
        return np.identity(3) << u.dimensionless_unscaled

    @property
    @abc.abstractmethod
    def translation_eff(self) -> u.Quantity:
        return vector.from_components() << u.mm

    def __call__(
            self,
            value: u.Quantity,
            rotate: bool = True,
            translate: bool = True,
            num_extra_dims: int = 0,
    ) -> u.Quantity:
        if rotate:
            sl = tuple([Ellipsis] + ([None] * num_extra_dims) + [slice(None), slice(None)])
            value = vector.matmul(self.rotation_eff[sl], value)
        if translate:
            sl = tuple([Ellipsis] + ([None] * num_extra_dims) + [slice(None)])
            value = value + self.translation_eff[sl]
        return value

    @abc.abstractmethod
    def __invert__(self) -> 'Transform':
        pass
