import typing as typ
import abc
import astropy.units as u
from kgpy import vector
from .. import Transform as BaseTransform

__all__ = ['Transform']


class Transform(BaseTransform):

    @property
    @abc.abstractmethod
    def rotation_eff(self) -> typ.Optional[u.Quantity]:
        return None

    @property
    @abc.abstractmethod
    def translation_eff(self) -> typ.Optional[u.Quantity]:
        return None

    def __call__(
            self,
            value: typ.Optional[u.Quantity] = None,
            rotate: bool = True,
            translate: bool = True,
            num_extra_dims: int = 0,
    ) -> u.Quantity:
        if value is not None:
            if rotate:
                if self.rotation_eff is not None:
                    sl = tuple([Ellipsis] + ([None] * num_extra_dims) + [slice(None), slice(None)])
                    value = vector.matmul(self.rotation_eff[sl], value)
        if translate:
            if self.translation_eff is not None:
                sl = tuple([Ellipsis] + ([None] * num_extra_dims) + [slice(None)])
                if value is None:
                    value = self.translation_eff[sl]
                else:
                    value = value + self.translation_eff[sl]
        return value

    @abc.abstractmethod
    def __invert__(self) -> 'Transform':
        pass

    @property
    def inverse(self) -> 'Transform':
        return self.__invert__()
