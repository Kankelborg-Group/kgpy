from typing import Self, TypeVar, Generic
import abc
import dataclasses
import kgpy.uncertainty
import kgpy.vectors
import kgpy.function
from . import vectors
from . import systems

__all__ = [
    'Instrument',
]

PointingT = TypeVar('PointingT', bound=kgpy.function.AbstractArray[kgpy.uncertainty.ArrayLike, vectors.FieldVector])
SystemT = TypeVar('SystemT', bound=systems.AbstractSystem)


@dataclasses.dataclass
class AbstractInstrument(
    abc.ABC,
    Generic[PointingT, SystemT],
):

    @property
    @abc.abstractmethod
    def pointing(self: Self) -> PointingT:
        pass

    @property
    @abc.abstractmethod
    def system(self: Self) -> SystemT:
        pass

    def __call__(
            self: Self,
            scene: kgpy.function.AbstractArray[vectors.TemporalSpectralFieldVector, kgpy.uncertainty.ArrayLike],
    ) -> kgpy.function.Array[vectors.TemporalSpectralPositionVector, kgpy.uncertainty.ArrayLike]:

        pointing = self.pointing(scene.input.time)

        images = self.system(
            scene=kgpy.function.Array(
                input=vectors.SpectralFieldVector(
                    wavelength=scene.input.wavelength,
                    field_x=scene.input.field_x - pointing.field_x,
                    field_y=scene.input.field_y - pointing.field_y,
                ),
                output=scene.output
            )
        )

        return kgpy.function.Array(
            input=vectors.TemporalSpectralPositionVector(
                time=scene.input.time,
                wavelength=images.input.wavelength,
                position_x=images.input.position_x,
                position_y=images.input.position_y,
            ),
            output=images.output,
        )

    def inverse(
            self: Self,
            image: kgpy.function.AbstractArray[vectors.TemporalSpectralPositionVector, kgpy.uncertainty.ArrayLike],
    ) -> kgpy.function.Array[vectors.TemporalSpectralFieldVector, kgpy.uncertainty.ArrayLike]:

        scene = self.system.inverse(
            images=kgpy.function.Array(
                input=vectors.SpectralPositionVector(
                    wavelength=image.input.wavelength,
                    position_x=image.input.position_x,
                    position_y=image.input.position_y,
                ),
                output=image.output,
            )
        )

        pointing = self.pointing(image.input.time)

        return kgpy.function.Array(
            input=vectors.TemporalSpectralFieldVector(
                time=image.input.time,
                wavelength=scene.input.wavelength,
                field_x=scene.input.field_x + pointing.field_x,
                field_y=scene.input.field_y + pointing.field_y,
            ),
            output=scene.output,
        )


@dataclasses.dataclass
class Instrument(
    AbstractInstrument[PointingT, SystemT],
):
    pointing: PointingT
    system: SystemT
