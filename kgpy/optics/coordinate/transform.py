import typing as typ
import abc
import numpy as np
import astropy.units as u
import kgpy.mixin
import kgpy.vector

__all__ = ['Transform']


class Transform(
    kgpy.mixin.Copyable,
    kgpy.mixin.Broadcastable,
    abc.ABC,
):
    @property
    @abc.abstractmethod
    def rotation_eff(self) -> u.Quantity:
        return np.identity(3) << u.dimensionless_unscaled

    @property
    @abc.abstractmethod
    def translation_eff(self) -> u.Quantity:
        return kgpy.vector.from_components() << u.mm

    def __call__(
            self,
            value: u.Quantity,
            use_rotations: bool = True,
            use_translations: bool = True,
            num_extra_dims: int = 0,
    ) -> u.Quantity:
        if use_rotations:
            sl = tuple([Ellipsis] + ([None] * num_extra_dims) + [slice(None), slice(None)])
            value = kgpy.vector.matmul(self.rotation_eff[sl], value)
        if use_translations:
            sl = tuple([Ellipsis] + ([None] * num_extra_dims) + [slice(None)])
            value = value + self.translation_eff[sl]
        return value

    @abc.abstractmethod
    def __invert__(self) -> 'Transform':
        pass
