import abc
import typing as typ
import dataclasses
import astropy.units as u
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vectors
import kgpy.function
from . import vectors
import numpy as np
import kgpy.matrix
from . import matrix
from . import apertures

PointSpreadT = typ.TypeVar('PointSpreadT', bound='PointSpread')
DistortionT = typ.TypeVar('DistortionT', bound='Distortion')
VignettingT = typ.TypeVar('VignettingT', bound='Vignetting')
EffectiveAreaT = typ.TypeVar('EffectiveAreaT', bound='EffectiveArea')
FieldStopT = typ.TypeVar('FieldStopT', bound='FieldStop')
AberrationT = typ.TypeVar('AberrationT', bound='Aberration')


@dataclasses.dataclass(eq=False)
class PointSpread(
):
    function: kgpy.function.AbstractArray[vectors.SpotVector, kgpy.uncertainty.ArrayLike]


@dataclasses.dataclass(eq=False)
class AbstractDistortion:

    @property
    @abc.abstractmethod
    def function(self) -> kgpy.function.AbstractArray[vectors.SpectralFieldVector, vectors.SpectralPositionVector]:
        pass

    def __call__(
            self: DistortionT,
            scene: kgpy.function.Array[vectors.OffsetSpectralFieldVector, kgpy.uncertainty.ArrayLike],
    ) -> kgpy.function.Array[vectors.SpectralPositionVector, kgpy.uncertainty.ArrayLike]:

        inputs = self.function(scene.input.vector_spectral_field)
        inputs.wavelength = scene.input.wavelength
        return kgpy.function.Array(
            input=inputs,
            output=scene.output,
        )

    def inverse(
            self: DistortionT,
            image: kgpy.function.Array[vectors.OffsetSpectralPositionVector, kgpy.uncertainty.ArrayLike]
    ) -> kgpy.function.Array[vectors.SpectralFieldVector, kgpy.uncertainty.ArrayLike]:

        return kgpy.function.Array(
            input=self.function.inverse(image.input.vector_spectral_position),
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
class Distortion(AbstractDistortion):
    function: kgpy.function.AbstractArray[vectors.SpectralFieldVector, vectors.SpectralPositionVector] = None


@dataclasses.dataclass(eq=False)
class ParameterizedDistortion(AbstractDistortion):
    scale: vectors.PositionVector = None
    dispersion: kgpy.uncertainty.ArrayLike = None
    angle_dispersion: kgpy.uncertainty.ArrayLike = None
    wavelength_reference: kgpy.uncertainty.ArrayLike = None
    detector_origin: vectors.PositionVector = None

    @property
    def function(self) -> kgpy.function.AbstractArray[vectors.SpectralFieldVector, vectors.SpectralPositionVector]:
        transform_vector = vectors.SpectralFieldVector(wavelength=self.wavelength_reference,
                                                       field_y=0*u.arcsec,
                                                       field_x=0*u.arcsec,
                                                       )

        rotation = matrix.SpectralFieldMatrix.rotation_spatial(self.angle_dispersion)

        scale = matrix.SpectralPositionMatrix(
            position_x=vectors.SpectralFieldVector(wavelength=self.dispersion, field_x=self.scale.position_x, field_y=0*self.scale.position_y.unit),
            position_y=vectors.SpectralFieldVector(wavelength=0*self.dispersion.unit, field_x=0*self.scale.position_x.unit, field_y=self.scale.position_y),
            wavelength=vectors.SpectralFieldVector(wavelength=1, field_x=0*self.scale.position_x.unit/self.dispersion.unit, field_y=0*self.scale.position_x.unit/self.dispersion.unit)
        )


        transform_matrix = scale @ rotation

        transform_vector = transform_matrix @ -transform_vector
        transform_vector = transform_vector - self.detector_origin

        transform_matrix = transform_matrix.transpose

        result = kgpy.function.Polynomial(
            input=None,
            coefficients=kgpy.vectors.CartesianND({
                '': transform_vector.position,
                'wavelength': transform_matrix.wavelength,
                'field_x': transform_matrix.field_x,
                'field_y': transform_matrix.field_y,
            })
        )

        return result


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
    function: kgpy.function.AbstractArray[kgpy.labeled.Array, kgpy.uncertainty.ArrayLike]

    def __call__(
            self: EffectiveAreaT,
            scene: kgpy.function.Array[vectors.SpectralFieldVector, kgpy.uncertainty.ArrayLike],
    ) -> kgpy.function.Array[vectors.SpectralFieldVector, kgpy.uncertainty.ArrayLike]:

        result = scene.copy_shallow()
        result.output = result.output * self.function(scene.input.wavelength)

        return result

    def inverse(
            self: EffectiveAreaT,
            scene: kgpy.function.Array[vectors.SpectralFieldVector, kgpy.uncertainty.ArrayLike],
    ) -> kgpy.function.Array[vectors.SpectralFieldVector, kgpy.uncertainty.ArrayLike]:

        result = scene.copy_shallow()
        result.output = result.output / self.function(scene.input.wavelength)

        return result


@dataclasses.dataclass(eq=False)
class FieldStop:

    aperture: apertures.Aperture

    def __call__(
            self: FieldStopT,
            scene: kgpy.function.Array[vectors.OffsetSpectralFieldVector, kgpy.uncertainty.ArrayLike],
    ) -> kgpy.function.Array[vectors.OffsetSpectralFieldVector, kgpy.uncertainty.ArrayLike]:

        where_unvignetted = ~self.aperture.is_unvignetted(scene.input.field_xy)

        result = scene.copy_shallow()
        result.output = result.output.copy()
        where_unvignetted = where_unvignetted.broadcast_to(result.output.shape)
        result.output[where_unvignetted] = 0

        return result

    def inverse(
            self: FieldStopT,
            scene: kgpy.function.Array[vectors.OffsetSpectralFieldVector, kgpy.uncertainty.ArrayLike],
    ) -> kgpy.function.Array[vectors.OffsetSpectralFieldVector, kgpy.uncertainty.ArrayLike]:
        return scene


# @dataclasses.dataclass(eq=False)
# class Aberration:
#
#     distortion: Distortion
#     point_spread: typ.Optional[PointSpread] = None
#     vignetting: typ.Optional[Vignetting] = None
#     effective_area: typ.Optional[EffectiveArea] = None
#     field_stop: typ.Optional[FieldStop] = None
#
#     def __call__(
#             self: AberrationT,
#             scene: kgpy.function.Array[vectors.SpectralFieldVector, kgpy.uncertainty.ArrayLike],
#     ) -> kgpy.function.Array[vectors.SpectralPositionVector, kgpy.uncertainty.ArrayLike]:
#
#         if self.effective_area is not None:
#             scene = self.effective_area(scene)
#
#         if self.field_stop is not None:
#             scene = self.field_stop(scene)
#
#         if self.vignetting is not None:
#             scene = self.vignetting(scene)
#
#         image = self.distortion(scene)
#
#         # if self.point_spread is not None:
#         #     result = self.point_spread(image)
#
#         return image
#
#     def inverse(
#             self: AberrationT,
#             image: kgpy.function.Array[vectors.SpectralPositionVector, kgpy.uncertainty.ArrayLike],
#     ) -> kgpy.function.Array[vectors.SpectralFieldVector, kgpy.uncertainty.ArrayLike]:
#
#         scene = self.distortion.inverse(image)
#
#         if self.vignetting is not None:
#             scene = self.vignetting.inverse(scene)
#
#         if self.field_stop is not None:
#             scene = self.field_stop.inverse(scene)
#
#         if self.effective_area is not None:
#             scene = self.effective_area.inverse(scene)
#
#         return scene
