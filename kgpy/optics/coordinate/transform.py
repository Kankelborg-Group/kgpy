import typing as typ
import abc
import astropy.units as u
import kgpy.mixin
import kgpy.optics

__all__ = ['Transform']


class Transform(
    kgpy.mixin.Copyable,
    kgpy.mixin.Broadcastable,
    abc.ABC,
):
    @abc.abstractmethod
    def __call__(
            self,
            value: u.Quantity,
            use_rotations: bool = True,
            use_translations: bool = True,
            num_extra_dims: int = 0,
    ) -> u.Quantity:
        pass

    @abc.abstractmethod
    def __invert__(self) -> 'Transform':
        pass
