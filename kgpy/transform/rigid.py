"""
Convenience classes for rigid transformations such as rotations and translations.
"""
import typing as typ
import abc
import collections
import dataclasses
import numpy as np
import astropy.units as u
from kgpy import mixin, vector, matrix, units
from . import Transform as BaseTransform

__all__ = [
    'Transform',
    'TransformList',
    'Transformable',
    'TiltX', 'TiltY', 'TiltZ',
    'Translate'
]


class Transform(BaseTransform):

    # @property
    # @abc.abstractmethod
    # def rotation_eff(self) -> typ.Optional[u.Quantity]:
    #     return None
    #
    # @property
    # @abc.abstractmethod
    # def translation_eff(self) -> typ.Optional[u.Quantity]:
    #     return None
    #
    # def __call__(
    #         self,
    #         value: typ.Optional[u.Quantity] = None,
    #         rotate: bool = True,
    #         translate: bool = True,
    #         num_extra_dims: int = 0,
    # ) -> u.Quantity:
    #     if value is not None:
    #         if rotate:
    #             if self.rotation_eff is not None:
    #                 sl = tuple([Ellipsis] + ([None] * num_extra_dims) + [slice(None), slice(None)])
    #                 value = vector.matmul(self.rotation_eff[sl], value)
    #     if translate:
    #         if self.translation_eff is not None:
    #             sl = tuple([Ellipsis] + ([None] * num_extra_dims) + [slice(None)])
    #             if value is None:
    #                 value = self.translation_eff[sl]
    #             else:
    #                 value = value + self.translation_eff[sl]
    #     return value

    @abc.abstractmethod
    def __call__(
            self,
            value: u.Quantity,
            rotate: bool = True,
            translate: bool = True,
            num_extra_dims: int = 0,
    ) -> u.Quantity:
        pass

    @abc.abstractmethod
    def __invert__(self) -> 'Transform':
        pass

    @property
    def inverse(self) -> 'Transform':
        return self.__invert__()


@dataclasses.dataclass
class TransformList(
    Transform,
    mixin.DataclassList[Transform],
):
    intrinsic: bool = True

    @property
    def transforms(self) -> typ.Iterator[Transform]:
        if self.intrinsic:
            return reversed(list(super().__iter__()))
        else:
            return super().__iter__()

    @property
    def extrinsic(self) -> bool:
        return not self.intrinsic

    # @property
    # def rotation_eff(self) -> u.Quantity:
    #     rotation = np.identity(3) << u.dimensionless_unscaled
    #     for transform in self.transforms:
    #         if transform is not None:
    #             if transform.rotation_eff is not None:
    #                 # rotation = matrix.mul(transform.rotation_eff, rotation)
    #                 rotation = matrix.mul(rotation, transform.rotation_eff)
    #     return rotation
    #
    @property
    def translation_eff(self) -> u.Quantity:
        value = vector.from_components() * u.mm
        return self(value)

    def __call__(
            self,
            value: u.Quantity,
            rotate: bool = True,
            translate: bool = True,
            num_extra_dims: int = 0,
    ) -> u.Quantity:
        for transform in self.transforms:
            value = transform(
                value=value,
                rotate=rotate,
                translate=translate,
                num_extra_dims=num_extra_dims,
            )
        return value

    def __invert__(self) -> 'TransformList':
        other = self.copy()
        other.data = []
        for transform in self:
            if transform is not None:
                transform = transform.__invert__()
            other.append(transform)
        other.reverse()
        return other

    # @property
    # def tol_iter(self) -> typ.Iterator['TransformList']:
    #     if len(self) > 0:
    #         for t in self[0].tol_iter:
    #             tl = type(self)([t], intrinsic=self.intrinsic)
    #             if len(self) > 1:
    #                 for tlist in self[1:].tol_iter:
    #                     yield tl + tlist
    #             else:
    #                 yield tl
    #     else:
    #         yield self.copy()

    def view(self) -> 'TransformList':
        other = super().view()      # type: TransformList
        other.intrinsic = self.intrinsic
        return other

    def copy(self) -> 'TransformList':
        other = super().copy()     # type: TransformList
        other.intrinsic = self.intrinsic
        return other


@dataclasses.dataclass
class Transformable(
    mixin.Toleranceable,
    mixin.Copyable,
):
    transform: TransformList = dataclasses.field(default_factory=TransformList)

    @property
    def tol_iter(self) -> typ.Iterator['Transformable']:
        others = super().tol_iter   # type: typ.Iterator[Transformable]
        for other in others:
            for transform in self.transform.tol_iter:
                new_other = other.view()
                new_other.transform = transform
                yield new_other

    def view(self) -> 'Transformable':
        other = super().view()  # type: Transformable
        other.transform = self.transform
        return other

    def copy(self) -> 'Transformable':
        other = super().copy()  # type: Transformable
        other.transform = self.transform.copy()
        return other


@dataclasses.dataclass
class TiltAboutAxis(Transform, abc.ABC):
    angle: typ.Union[u.Quantity, units.TolQuantity] = 0 * u.deg

    @property
    def broadcasted(self):
        return np.broadcast(super().broadcasted, self.angle)

    def __eq__(self, other: 'TiltAboutAxis') -> bool:
        return np.array(self.angle == other.angle).all()

    def __call__(
            self,
            value: u.Quantity,
            rotate: bool = True,
            translate: bool = True,
            num_extra_dims: int = 0,
    ) -> u.Quantity:
        if rotate:
            sl = tuple([Ellipsis] + [None] * num_extra_dims + [slice(None), slice(None)])
            value = vector.matmul(self.rotation_matrix[sl], value)
        return value

    def __invert__(self) -> 'TiltAboutAxis':
        return type(self)(angle=-self.angle)

    @property
    @abc.abstractmethod
    def rotation_matrix(self) -> u.Quantity:
        pass

    @property
    def tol_iter(self) -> typ.Iterator['TiltAboutAxis']:
        others = super().tol_iter   # type: typ.Iterator[TiltAboutAxis]
        for other in others:
            if isinstance(self.angle, units.TolQuantity):
                other_1, other_2 = other.view(), other.view()
                other_1.angle = self.angle.amin
                other_2.angle = self.angle.amax
                yield other_1
                yield other_2
            else:
                yield other

    def view(self) -> 'TiltAboutAxis':
        other = super().view()     # type: TiltAboutAxis
        other.angle = self.angle
        return other

    def copy(self) -> 'TiltAboutAxis':
        other = super().copy()  # type: TiltAboutAxis
        other.angle = self.angle.copy()
        return other


class TiltX(TiltAboutAxis):

    @property
    def rotation_matrix(self) -> u.Quantity:
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
    def rotation_matrix(self) -> u.Quantity:
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
    def rotation_matrix(self) -> u.Quantity:
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

    x: typ.Union[u.Quantity, units.TolQuantity] = 0 * u.mm
    y: typ.Union[u.Quantity, units.TolQuantity] = 0 * u.mm
    z: typ.Union[u.Quantity, units.TolQuantity] = 0 * u.mm

    @classmethod
    def from_vector(
            cls,
            vec: u.Quantity,
    ) -> 'Translate':
        return cls(
            x=vec[vector.x].copy(),
            y=vec[vector.y].copy(),
            z=vec[vector.z].copy(),
        )

    @property
    def broadcasted(self):
        return np.broadcast(
            super().broadcasted,
            self.x,
            self.y,
            self.z,
        )

    @property
    def vector(self):
        return vector.from_components(x=self.x, y=self.y, z=self.z)

    def __call__(
            self,
            value: u.Quantity,
            rotate: bool = True,
            translate: bool = True,
            num_extra_dims: int = 0,
    ) -> u.Quantity:
        if translate:
            sl = tuple([Ellipsis] + [None] * num_extra_dims + [slice(None)])
            value = self.vector[sl] + value
        return value

    def __invert__(self) -> 'Translate':
        return Translate(x=-self.x, y=-self.y, z=-self.z)

    def __eq__(self, other: 'Translate') -> bool:
        return np.array(self.vector == other.vector).all()

    def __add__(self, other: 'Translate') -> 'Translate':
        return type(self)(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z,
        )

    def __sub__(self, other: 'Translate') -> 'Translate':
        return type(self)(
            x=self.x - other.x,
            y=self.y - other.y,
            z=self.z - other.z,
        )

    # @property
    # def rotation_eff(self) -> None:
    #     return None
    #
    # @property
    # def translation_eff(self) -> u.Quantity:
    #     return self.vector

    @property
    def tol_iter(self):
        others = super().tol_iter   # type: typ.Iterator[Translate]

        if isinstance(self.x, units.TolQuantity):
            ax = [self.x.amin, self.x.amax]
        else:
            ax = [self.x.copy()]

        if isinstance(self.y, units.TolQuantity):
            ay = [self.y.amin, self.y.amax]
        else:
            ay = [self.y.copy()]

        if isinstance(self.z, units.TolQuantity):
            az = [self.z.amin, self.z.amax]
        else:
            az = [self.z.copy()]

        # if (len(ax) > 1) or (len(ay) > 1) or (len(az) > 1):

        for other in others:
            for ax_i in ax:
                for ay_j in ay:
                    for az_k in az:
                        new_other = other.view()
                        new_other.x = ax_i
                        new_other.y = ay_j
                        new_other.z = az_k
                        yield new_other

    def view(self) -> 'Translate':
        other = super().view()  # type: Translate
        other.x = self.x
        other.y = self.y
        other.z = self.z
        return other

    def copy(self) -> 'Translate':
        other = super().copy()  # type: Translate
        other.x = self.x.copy()
        other.y = self.y.copy()
        other.z = self.z.copy()
        return other
