import typing as typ
import dataclasses
import collections
import numpy as np
import astropy.units as u
from kgpy import vector, matrix
from . import Transform

__all__ = ['TransformList']

TransformT = typ.TypeVar('TransformT', bound=Transform)


class TransformList(
    Transform,
    collections.UserList,
):

    @property
    def extrinsic_transforms(self):
        return reversed(self)

    @property
    def rotation_eff(self) -> u.Quantity:
        rotation = super().rotation_eff
        for transform in self.extrinsic_transforms:
            if transform is not None:
                rotation = matrix.mul(transform.rotation_eff, rotation)
        return rotation

    @property
    def translation_eff(self) -> u.Quantity:
        translation = super().translation_eff
        for transform in self.extrinsic_transforms:
            if transform is not None:
                translation = vector.matmul(transform.rotation_eff, translation) + transform.translation_eff
        return translation

    def __invert__(self) -> 'TransformList':
        other = type(self)()
        for transform in self:
            if transform is not None:
                transform = transform.__invert__()
            other.append(transform)
        other.reverse()
        return other

    def copy(self) -> 'TransformList':
        other = type(self)()
        for transform in self:
            if transform is not None:
                transform = transform.copy()
            other.append(transform)
        return other

