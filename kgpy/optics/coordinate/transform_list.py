import typing as typ
import dataclasses
import collections
import astropy.units as u
from . import Transform

__all__ = ['TransformList']

TransformT = typ.TypeVar('TransformT', bound=Transform)


class TransformList(
    Transform,
    collections.UserList,
    typ.Generic[TransformT],
):

    def __invert__(self) -> 'TransformList':
        other = type(self)()
        for transform in self:
            if transform is not None:
                transform = transform.__invert__()
            other.append(transform)
        other.reverse()
        return other

    def __call__(
            self,
            value: u.Quantity,
            use_rotations: bool = True,
            use_translations: bool = True,
            num_extra_dims: int = 0,
    ) -> u.Quantity:
        value = value.copy()
        for transform in reversed(self):
            if transform is not None:
                value = transform(value, use_rotations, use_translations, num_extra_dims)
        return value

    def copy(self) -> 'TransformList':
        other = type(self)()
        for transform in self:
            if transform is not None:
                transform = transform.copy()
            other.append(transform)
        return other

