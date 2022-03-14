import dataclasses
import typing as typ
import abc
import astropy.units as u
import kgpy.mixin
import kgpy.uncertainty
import kgpy.vectors
import kgpy.matrix

__all__ = [
    'AbstractTransform',
    'Translation',
    'AbstractRotation',
    'RotationX',
    'RotationY',
    'RotationZ',
    'TransformList',
    'Transformable',
]

AbstractTransformT = typ.TypeVar('AbstractTransformT', bound='AbstractTransform')
TranslationT = typ.TypeVar('TranslationT', bound='Translation')
AbstractRotationT = typ.TypeVar('AbstractRotationT', bound='AbstractRotation')
RotationXT = typ.TypeVar('RotationXT', bound='RotationX')
RotationYT = typ.TypeVar('RotationYT', bound='RotationY')
RotationZT = typ.TypeVar('RotationZT', bound='RotationZ')
TransformListT = typ.TypeVar('TransformListT', bound='TransformList')


class AbstractTransform(
    abc.ABC,
):

    @property
    def matrix(self: AbstractTransformT) -> kgpy.matrix.Cartesian3D:
        return kgpy.matrix.Cartesian3D.identity()

    @property
    def vector(self: AbstractTransformT) -> kgpy.vectors.Cartesian3D:
        return kgpy.vectors.Cartesian3D() * u.mm

    def __call__(
            self: AbstractTransformT,
            value: kgpy.vectors.Cartesian3D,
            rotate: bool = True,
            translate: bool = True,
    ) -> kgpy.vectors.Cartesian3D:
        if rotate:
            value = self.matrix @ value
        if translate:
            value = value + self.vector
        return value

    @abc.abstractmethod
    def __invert__(self: AbstractTransformT) -> AbstractTransformT:
        pass

    @property
    def inverse(self: AbstractTransformT) -> AbstractTransformT:
        return self.__invert__()


@dataclasses.dataclass
class Translation(AbstractTransform):
    vector: kgpy.vectors.Cartesian3D = None

    def __post_init__(self: TranslationT) -> None:
        if self.vector is None:
            self.vector = kgpy.vectors.Cartesian3D() * u.mm

    def __invert__(self: TranslationT) -> TranslationT:
        return type(self)(vector=-self.vector)


@dataclasses.dataclass
class AbstractRotation(AbstractTransform):
    angle: kgpy.uncertainty.ArrayLike

    def __invert__(self: AbstractRotationT) -> AbstractRotationT:
        return type(self)(angle=-self.angle)


@dataclasses.dataclass
class RotationX(AbstractRotation):

    @property
    def matrix(self: RotationXT) -> kgpy.matrix.Cartesian3D:
        return kgpy.matrix.Cartesian3D.rotation_x(self.angle)


@dataclasses.dataclass
class RotationY(AbstractRotation):

    @property
    def matrix(self: RotationYT) -> kgpy.matrix.Cartesian3D:
        return kgpy.matrix.Cartesian3D.rotation_y(self.angle)


@dataclasses.dataclass
class RotationZ(AbstractRotation):

    @property
    def matrix(self: RotationZT) -> kgpy.matrix.Cartesian3D:
        return kgpy.matrix.Cartesian3D.rotation_z(self.angle)


@dataclasses.dataclass
class TransformList(
    AbstractTransform,
    kgpy.mixin.DataclassList,
):

    intrinsic: bool = True

    @property
    def extrinsic(self: TransformListT) -> bool:
        return not self.intrinsic

    @property
    def transforms(self: TransformListT) -> typ.Iterator[AbstractTransform]:
        if self.intrinsic:
            return reversed(list(self))
        else:
            return iter(self)

    @property
    def matrix(self: TransformListT) -> kgpy.matrix.Cartesian3D:
        rotation = kgpy.matrix.Cartesian3D.identity()

        for transform in reversed(list(self.transforms)):
            if transform is not None:
                rotation = rotation @ transform.matrix

        return rotation

    @property
    def vector(self: TransformListT) -> kgpy.vectors.Cartesian3D:
        rotation = kgpy.matrix.Cartesian3D.identity()
        translation = kgpy.vectors.Cartesian3D() * u.mm

        for transform in reversed(list(self.transforms)):
            if transform is not None:
                rotation = rotation @ transform.matrix
                translation = rotation @ transform.vector + translation

        return translation

    def __invert__(self: TransformListT) -> TransformListT:
        other = self.copy()
        other.data = []
        for transform in self:
            if transform is not None:
                transform = transform.__invert__()
            other.append(transform)
        other.reverse()
        return other


@dataclasses.dataclass
class Transformable:
    transform: TransformList = dataclasses.field(default_factory=TransformList)
