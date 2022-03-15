import typing as typ
import dataclasses
import astropy.units as u
import kgpy.uncertainty
import kgpy.vectors
import kgpy.function
from . import vectors

PointSpreadFunctionT = typ.TypeVar('PointSpreadFunctionT', bound='PointSpreadFunction')
DistortionT = typ.TypeVar('DistortionT', bound='Distortion')


@dataclasses.dataclass(eq=False)
class PointSpreadFunction(
    kgpy.function.Array[vectors.SpotVector, kgpy.uncertainty.ArrayLike]
):
    input: vectors.SpotVector = dataclasses.field(default_factory=vectors.SpotVector)
    output: kgpy.uncertainty.ArrayLike = 0 * u.dimensionless_unscaled


@dataclasses.dataclass
class DistortionFunction(
    kgpy.function.PolynomialArray[vectors.FieldVector, vectors.FieldVector],
):
    @property
    def plate_scale(self: DistortionT) -> kgpy.vectors.Cartesian2D:
        axis = ('field_x', 'field_y')
        return self.input.field.ptp(axis=axis).length / self.output.field.ptp(axis=axis)

    @property
    def dispersion(self):
        axis = ('wavelength', 'velocity_los')
        return self.input.wavelength.ptp(axis=axis) / self.output.position.ptp(axis=axis)


@dataclasses.dataclass(eq=False)
class VignettingFunction(
    kgpy.function.PolynomialArray[vectors.FieldVector, kgpy.uncertainty.ArrayLike]
):
    pass
