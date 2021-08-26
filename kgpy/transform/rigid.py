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

    @abc.abstractmethod
    def __call__(
            self,
            value: vector.Vector3D,
            rotate: bool = True,
            translate: bool = True,
            num_extra_dims: int = 0,
    ) -> vector.Vector3D:
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

    @property
    def rotation_eff(self) -> u.Quantity:
        rotation = matrix.Matrix3D(
            xx=1 * u.dimensionless_unscaled,
            yy=1 * u.dimensionless_unscaled,
            zz=1 * u.dimensionless_unscaled,
        )
        for transform in self.transforms:
            if transform is not None:
                if isinstance(transform, TiltAboutAxis):
                    rotation = transform.rotation_matrix @ rotation
        return rotation

    @property
    def translation_eff(self) -> vector.Vector3D:
        value = vector.Vector3D.spatial()
        return self(value)

    def __call__(
            self,
            value: vector.Vector3D,
            rotate: bool = True,
            translate: bool = True,
            num_extra_dims: int = 0,
    ) -> vector.Vector3D:

        rotation = matrix.Matrix3D(
            xx=1 * u.dimensionless_unscaled,
            yy=1 * u.dimensionless_unscaled,
            zz=1 * u.dimensionless_unscaled,
        )
        translation = vector.Vector3D(x=0 * value.x.unit, y=0 * value.y.unit, z=0 * value.z.unit)

        for transform in reversed(list(self.transforms)):
            if transform is not None:
                if isinstance(transform, TiltAboutAxis):
                    if rotate:
                        rotation = rotation @ transform.rotation_matrix
                elif isinstance(transform, Translate):
                    if translate:
                        translation = (rotation @ transform.value) + translation

        extra_dims_slice = (Ellipsis, ) + num_extra_dims * (np.newaxis, )
        return (rotation[extra_dims_slice] @ value) + translation[extra_dims_slice]

        # for transform in self.transforms:
        #     value = transform(
        #         value=value,
        #         rotate=rotate,
        #         translate=translate,
        #         num_extra_dims=num_extra_dims,
        #     )
        # return value


    def __invert__(self) -> 'TransformList':
        other = self.copy()
        other.data = []
        for transform in self:
            if transform is not None:
                transform = transform.__invert__()
            other.append(transform)
        other.reverse()
        return other

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
class Tilt(Transform, abc.ABC):

    @property
    @abc.abstractmethod
    def rotation_matrix(self) -> matrix.Matrix3D:
        pass

    def __call__(
            self,
            value: vector.Vector3D,
            rotate: bool = True,
            translate: bool = True,
            num_extra_dims: int = 0,
    ) -> vector.Vector3D:
        if rotate:
            extra_dims_slice = (Ellipsis, ) + num_extra_dims * (np.newaxis, )
            value = self.rotation_matrix[extra_dims_slice] @ value
        return value


@dataclasses.dataclass
class TiltGeneral(Tilt):
    rotation: matrix.Matrix3D = dataclasses.field(default_factory=matrix.Matrix3D)

    @property
    def rotation_matrix(self) -> matrix.Matrix3D:
        return self.rotation

    def __eq__(self, other: 'TiltGeneral') -> bool:
        return self.rotation_matrix == other.rotation_matrix

    def __invert__(self) -> 'TiltGeneral':
        return type(self)(rotation=self.rotation_matrix.transpose)

    def view(self) -> 'TiltGeneral':
        other = super().view()     # type: TiltGeneral
        other.rotation = self.rotation
        return other

    def copy(self) -> 'TiltGeneral':
        other = super().copy()  # type: TiltGeneral
        other.rotation = self.rotation.copy()
        return other


@dataclasses.dataclass
class TiltAboutAxis(Tilt, abc.ABC):
    angle: typ.Union[u.Quantity, units.TolQuantity] = 0 * u.deg

    @property
    def broadcasted(self):
        return np.broadcast(super().broadcasted, self.angle)

    def __eq__(self, other: 'TiltAboutAxis') -> bool:
        return np.array(self.angle == other.angle).all()

    def __invert__(self) -> 'TiltAboutAxis':
        return type(self)(angle=-self.angle)

    @property
    @abc.abstractmethod
    def rotation_matrix(self) -> matrix.Matrix3D:
        pass

    @property
    def tol_iter(self) -> typ.Iterator['TiltAboutAxis']:
        others = super().tol_iter   # type: typ.Iterator[TiltAboutAxis]
        for other in others:
            if isinstance(self.angle, units.TolQuantity):
                other_1, other_2 = other.view(), other.view()
                other_1.angle = self.angle.vmin
                other_2.angle = self.angle.vmax
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
    def rotation_matrix(self) -> matrix.Matrix3D:
        r = matrix.Matrix3D()
        cos_angle, sin_angle = np.cos(self.angle), np.sin(self.angle)
        r.xx = 1
        r.yy = cos_angle
        r.yz = -sin_angle
        r.zy = sin_angle
        r.zz = cos_angle
        return r


class TiltY(TiltAboutAxis):

    @property
    def rotation_matrix(self) -> matrix.Matrix3D:
        r = matrix.Matrix3D()
        cos_angle, sin_angle = np.cos(self.angle), np.sin(self.angle)
        r.xx = cos_angle
        r.xz = sin_angle
        r.yy = 1
        r.zx = -sin_angle
        r.zz = cos_angle
        return r


class TiltZ(TiltAboutAxis):

    @property
    def rotation_matrix(self) -> matrix.Matrix3D:
        r = matrix.Matrix3D()
        cos_angle, sin_angle = np.cos(self.angle), np.sin(self.angle)
        r.xx = cos_angle
        r.xy = -sin_angle
        r.yx = sin_angle
        r.yy = cos_angle
        r.zz = 1
        return r


@dataclasses.dataclass
class Translate(Transform):

    x: typ.Union[u.Quantity, units.TolQuantity] = 0 * u.mm
    y: typ.Union[u.Quantity, units.TolQuantity] = 0 * u.mm
    z: typ.Union[u.Quantity, units.TolQuantity] = 0 * u.mm

    @classmethod
    def from_vector(
            cls,
            vec: vector.Vector3D,
    ) -> 'Translate':
        return cls(
            x=vec.x.copy(),
            y=vec.y.copy(),
            z=vec.z.copy(),
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
    def value(self):
        return vector.Vector3D(x=self.x, y=self.y, z=self.z)

    def __call__(
            self,
            value: vector.Vector3D,
            rotate: bool = True,
            translate: bool = True,
            num_extra_dims: int = 0,
    ) -> vector.Vector3D:
        if translate:
            extra_dims_slice = (Ellipsis, ) + num_extra_dims * (np.newaxis, )
            value = self.value[extra_dims_slice] + value
        return value

    def __invert__(self) -> 'Translate':
        return Translate(x=-self.x, y=-self.y, z=-self.z)

    def __eq__(self, other: 'Translate') -> bool:
        return np.array(self.value == other.value).all()

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

    @property
    def tol_iter(self):
        others = super().tol_iter   # type: typ.Iterator[Translate]

        if isinstance(self.x, units.TolQuantity):
            ax = [self.x.vmin, self.x.vmax]
        else:
            ax = [self.x.copy()]

        if isinstance(self.y, units.TolQuantity):
            ay = [self.y.vmin, self.y.vmax]
        else:
            ay = [self.y.copy()]

        if isinstance(self.z, units.TolQuantity):
            az = [self.z.vmin, self.z.vmax]
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
