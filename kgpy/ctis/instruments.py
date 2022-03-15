import typing as typ
import abc
import dataclasses
import kgpy.uncertainty
import kgpy.function
import kgpy.optics
from . import vectors

__all__ = [
    'AbstractInstrument',
    'AbstractAberrationInstrument',
    'AberrationInstrument',
]

SceneT = kgpy.function.AbstractArray[kgpy.optics.vectors.FieldVector, kgpy.uncertainty.ArrayLike]
ImageT = kgpy.function.AbstractArray[vectors.PixelVector, kgpy.uncertainty.ArrayLike]
AbstractInstrumentT = typ.TypeVar('AbstractInstrumentT', bound='AbstractInstrument')
AbstractAberrationInstrumentT = typ.TypeVar('AbstractAberrationInstrumentT', bound='AbstractAberrationInstrument')


@dataclasses.dataclass
class AbstractInstrument(
    abc.ABC,
):

    @abc.abstractmethod
    def __call__(self: AbstractInstrumentT, scene: SceneT) -> ImageT:
        pass

    @abc.abstractmethod
    def deproject(self: AbstractInstrumentT, image: ImageT) -> SceneT:
        pass


@dataclasses.dataclass
class AbstractAberrationInstrument(
    AbstractInstrument,
):

    @property
    @abc.abstractmethod
    def aberration(self: AbstractAberrationInstrumentT) -> kgpy.optics.aberrations.Aberration:
        pass

    def __call__(self: AbstractAberrationInstrumentT, scene: SceneT) -> ImageT:
        return self.aberration(scene)

    def deproject(self: AbstractAberrationInstrumentT, image: ImageT) -> SceneT:
        return self.aberration.inverse(image)


@dataclasses.dataclass
class AberrationInstrument(
    AbstractAberrationInstrument
):

    aberration: kgpy.optics.aberrations.Aberration
