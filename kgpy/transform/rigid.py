"""
Convenience classes for rigid transformations such as rotations and translations.
"""
import typing as typ
import abc
import collections
import dataclasses
import numpy as np
import astropy.units as u
from kgpy import mixin, vector, matrix
from . import Transform as BaseTransform

__all__ = [
    'Transform',
    'TransformList',
    'Transformable',
    'TiltX', 'TiltY', 'TiltZ',
    'Translate'
]


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
        return rotation

    @property
    def translation_eff(self) -> u.Quantity:
        translation = None
        for transform in self.transforms:
            if transform is not None:
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


@dataclasses.dataclass
class TiltAboutAxis(Transform, abc.ABC):
    angle: u.Quantity = 0 * u.deg

    @property
    def broadcasted(self):
        return np.broadcast(super().broadcasted, self.angle)

    def __eq__(self, other: 'TiltAboutAxis') -> bool:
        return np.array(self.angle == other.angle).all()

    def __invert__(self) -> 'TiltAboutAxis':
        return type(self)(angle=-self.angle)

    @property
    def translation_eff(self) -> None:
        return None

    def copy(self) -> 'TiltAboutAxis':
        return type(self)(
            angle=self.angle.copy(),
        )


class TiltX(TiltAboutAxis):

    @property
    def rotation_eff(self) -> u.Quantity:
        r = np.zeros(self.shape + (3, 3)) << u.dimensionless_unscaled
        cos_x, sin_x = np.cos(self.angle), np.sin(self.angle)
        r[..., 0, 0] = 1
        r[..., 1, 1] = cos_x
        r[..., 1, 2] = -sin_x
        r[..., 2, 1] = sin_x
        r[..., 2, 2] = cos_x
        return r


class TiltY(TiltAboutAxis):

    @property
    def rotation_eff(self) -> u.Quantity:
        r = np.zeros(self.shape + (3, 3)) << u.dimensionless_unscaled
        cos_y, sin_y = np.cos(self.angle), np.sin(self.angle)
        r[..., 0, 0] = cos_y
        r[..., 0, 2] = sin_y
        r[..., 1, 1] = 1
        r[..., 2, 0] = -sin_y
        r[..., 2, 2] = cos_y
        return r


class TiltZ(TiltAboutAxis):

    @property
    def rotation_eff(self) -> u.Quantity:
        r = np.zeros(self.shape + (3, 3)) << u.dimensionless_unscaled
        cos_z, sin_z = np.cos(self.angle), np.sin(self.angle)
        r[..., 0, 0] = cos_z
        r[..., 0, 1] = -sin_z
        r[..., 1, 0] = sin_z
        r[..., 1, 1] = cos_z
        r[..., 2, 2] = 1
        return r


@dataclasses.dataclass
class Translate(Transform):

    vector: u.Quantity = dataclasses.field(default_factory=vector.from_components)

    @classmethod
    def from_components(cls, x: u.Quantity = 0 * u.mm, y: u.Quantity = 0 * u.mm, z: u.Quantity = 0 * u.mm):
        return cls(vector=vector.from_components(x, y, z))

    @property
    def broadcasted(self):
        return np.broadcast(
            super().broadcasted,
            self.vector,
        )

    @property
    def x(self) -> u.Quantity:
        return self.vector[vector.x]

    @x.setter
    def x(self, value: u.Quantity) -> typ.NoReturn:
        self.vector[vector.x] = value

    @property
    def y(self) -> u.Quantity:
        return self.vector[vector.y]

    @y.setter
    def y(self, value: u.Quantity) -> typ.NoReturn:
        self.vector[vector.y] = value

    @property
    def z(self) -> u.Quantity:
        return self.vector[vector.z]

    @z.setter
    def z(self, value: u.Quantity) -> typ.NoReturn:
        self.vector[vector.z] = value

    def __invert__(self) -> 'Translate':
        return Translate(-self.vector)

    def __eq__(self, other: 'Translate') -> bool:
        return np.array(self.vector == other.vector).all()

    def __add__(self, other: 'Translate') -> 'Translate':
        return type(self)(self.vector + other.vector)

    def __sub__(self, other: 'Translate') -> 'Translate':
        return type(self)(self.vector - other.vector)

    @property
    def rotation_eff(self) -> None:
        return None

    @property
    def translation_eff(self) -> u.Quantity:
        return self.vector

    def copy(self) -> 'Translate':
        return Translate(
            vector=self.vector.copy()
        )
