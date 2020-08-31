import typing as typ
import dataclasses
import collections
import numpy as np
import astropy.units as u
from kgpy import vector, matrix, mixin
from . import Transform

__all__ = ['TransformList', 'Transformable']


class TransformList(
    Transform,
    collections.UserList,
):
    def __init__(self, transforms: typ.List[Transform] = None, intrinsic: bool = True):
        super().__init__(transforms)
        self.intrinsic = intrinsic

    def __repr__(self):
        return type(self).__name__ + '(transforms=' + super().__repr__() + ', intrinsic=' + str(self.intrinsic) + ')'

    @property
    def transforms(self) -> typ.Iterator[Transform]:
        if self.intrinsic:
            return reversed(list(super().__iter__()))
        else:
            return super().__iter__()

    @property
    def extrinsic(self) -> bool:
        return not self.intrinsic

    @property
    def rotation_eff(self) -> u.Quantity:
        rotation = np.identity(3) << u.dimensionless_unscaled
        for transform in self.transforms:
            if transform is not None:
                if transform.rotation_eff is not None:
                    rotation = matrix.mul(transform.rotation_eff, rotation)
                    # rotation = matrix.mul(rotation, transform.rotation_eff, )
        return rotation

    @property
    def translation_eff(self) -> u.Quantity:
        translation = None
        for transform in self.transforms:
            if transform is not None:
                # translation = vector.matmul(transform.rotation_eff, translation) + transform.translation_eff
                translation = transform(translation)
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
        other.intrinsic = self.intrinsic
        return other


@dataclasses.dataclass
class Transformable(
    mixin.Copyable,
):
    transform: TransformList = dataclasses.field(default_factory=TransformList)

    def copy(self) -> 'Transformable':
        other = super().copy()  # type: Transformable
        other.transform = self.transform.copy()
        return other
