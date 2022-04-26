import typing as typ
import dataclasses
import astropy.units as u
import kgpy.uncertainty
import kgpy.vectors
import kgpy.function
from . import vectors

PointSpreadT = typ.TypeVar('PointSpreadT', bound='PointSpread')
DistortionT = typ.TypeVar('DistortionT', bound='Distortion')
VignettingT = typ.TypeVar('VignettingT', bound='Vignetting')
AberrationT = typ.TypeVar('AberrationT', bound='Aberration')


@dataclasses.dataclass(eq=False)
class PointSpread(
):
    function: kgpy.function.AbstractArray[vectors.SpotVector, kgpy.uncertainty.ArrayLike]


@dataclasses.dataclass(eq=False)
class Distortion:
    function: kgpy.function.AbstractArray[vectors.SpectralFieldVector, vectors.SpectralPositionVector]

    def __call__(
            self: DistortionT,
            scene: kgpy.function.Array[vectors.SpectralFieldVector, kgpy.uncertainty.ArrayLike],
    ) -> kgpy.function.Array[vectors.SpectralPositionVector, kgpy.uncertainty.ArrayLike]:

        result_input = self.function(scene.input)

        result = kgpy.function.Array(
            input=vectors.SpectralPositionVector(
                wavelength=scene.input.wavelength,
                position_x=result_input.position_x,
                position_y=result_input.position_y
            ),
            output=scene.output,
        )

        return result

    def inverse(
            self: DistortionT,
            image: kgpy.function.Array[vectors.SpectralPositionVector, kgpy.uncertainty.ArrayLike]
    ) -> kgpy.function.Array[vectors.SpectralFieldVector, kgpy.uncertainty.ArrayLike]:

        result_input = self.function.inverse(image.input)

        return kgpy.function.Array(
            input=vectors.SpectralFieldVector(
                wavelength=image.input.wavelength,
                field_x=result_input.field_x,
                field_y=result_input.field_y,
            ),
            output=image.output,
        )

    @property
    def plate_scale(self: DistortionT) -> kgpy.vectors.Cartesian2D:
        axis = ('field_x', 'field_y')
        return self.function.input.field_xy.ptp(axis=axis).length / self.function.output.position_xy.ptp(axis=axis)

    @property
    def dispersion(self):
        axis = ('field_x', 'field_y', 'wavelength', 'velocity_los')
        return self.function.input.wavelength.ptp(axis=axis) / self.function.output.position_xy.ptp(axis=axis).length


@dataclasses.dataclass(eq=False)
class Vignetting:
    function: kgpy.function.AbstractArray[vectors.SpectralFieldVector, kgpy.uncertainty.ArrayLike]

    def __call__(
            self: VignettingT,
            scene: kgpy.function.Array[vectors.SpectralFieldVector, kgpy.uncertainty.ArrayLike],
    ) -> kgpy.function.Array[vectors.SpectralFieldVector, kgpy.uncertainty.ArrayLike]:

        return kgpy.function.Array(
            input=scene.input,
            output=scene.output * self.function(scene.input),
        )

    def inverse(
            self: VignettingT,
            scene: kgpy.function.Array[vectors.SpectralFieldVector, kgpy.uncertainty.ArrayLike],
    ) -> kgpy.function.Array[vectors.SpectralFieldVector, kgpy.uncertainty.ArrayLike]:

        return kgpy.function.Array(
            input=scene.input,
            output=scene.output / self.function(scene.input),
        )


@dataclasses.dataclass(eq=False)
class EffectiveArea:
    pass


@dataclasses.dataclass(eq=False)
class Aberration:

    # point_spread: PointSpread
    distortion: Distortion
    # vignetting: Vignetting
    # effective_area: EffectiveArea

    def __call__(
            self: AberrationT,
            scene: kgpy.function.Array[vectors.SpectralFieldVector, kgpy.uncertainty.ArrayLike],
    ) -> kgpy.function.Array[vectors.SpectralPositionVector, kgpy.uncertainty.ArrayLike]:

        # result = self.vignetting(scene)
        result = self.distortion(scene)

        return result

    def inverse(
            self: AberrationT,
            image: kgpy.function.Array[vectors.SpectralPositionVector, kgpy.uncertainty.ArrayLike],
    ) -> kgpy.function.Array[vectors.SpectralFieldVector, kgpy.uncertainty.ArrayLike]:

        result = self.distortion.inverse(image)
        result = self.vignetting.inverse(result)

        return result
