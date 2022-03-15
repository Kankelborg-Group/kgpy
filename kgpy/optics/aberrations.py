import typing as typ
import dataclasses
import astropy.units as u
import kgpy.uncertainty
import kgpy.vectors
import kgpy.function
from . import vectors

PointSpreadT = typ.TypeVar('PointSpreadT', bound='PointSpread')
DistortionT = typ.TypeVar('DistortionT', bound='Distortion')
VignettingT = typ.TypeVar('Vignetting', bound='Vignetting')
AberrationT = typ.TypeVar('AberrationT', bound='Aberration')


@dataclasses.dataclass(eq=False)
class PointSpread(
):
    function: kgpy.function.AbstractArray[vectors.SpotVector, kgpy.uncertainty.ArrayLike]


@dataclasses.dataclass(eq=False)
class Distortion:
    function: kgpy.function.AbstractArray[vectors.FieldVector, vectors.ImageVector]

    @property
    def plate_scale(self: DistortionT) -> kgpy.vectors.Cartesian2D:
        axis = ('field_x', 'field_y')
        return self.function.input.field.ptp(axis=axis).length / self.function.output.position.ptp(axis=axis)

    @property
    def dispersion(self):
        axis = ('wavelength', 'velocity_los')
        return self.function.input.wavelength.ptp(axis=axis) / self.function.output.position.ptp(axis=axis)


@dataclasses.dataclass(eq=False)
class Vignetting:
    function: kgpy.function.AbstractArray[vectors.FieldVector, kgpy.uncertainty.ArrayLike]


@dataclasses.dataclass(eq=False)
class Aberration:

    point_spread: PointSpread
    distortion: Distortion
    vignetting: Vignetting

    def __call__(
            self: AberrationT,
            scene: kgpy.function.Array[vectors.FieldVector, kgpy.uncertainty.ArrayLike],
    ) -> kgpy.function.Array[vectors.ImageVector, kgpy.uncertainty.ArrayLike]:

        raise NotImplementedError

    def inverse(
            self: AberrationT,
            image: kgpy.function.Array[vectors.ImageVector, kgpy.uncertainty.ArrayLike],
    ) -> kgpy.function.Array[vectors.FieldVector, kgpy.uncertainty.ArrayLike]:

        raise NotImplementedError
