import typing as typ
import dataclasses
import astropy.units as u
import astropy.constants
import kgpy.uncertainty
import kgpy.vectors
import kgpy.transforms

__all__ = [
    'StokesVector',
    'SpectralFieldVector',
    'ObjectVector',
    'SpotVector',
]

SphericalT = typ.TypeVar('SphericalT', bound='Spherical')
StokesVectorT = typ.TypeVar('StokesVectorT', bound='StokesVector')


@dataclasses.dataclass(eq=False)
class Spherical(
    kgpy.vectors.AbstractVector,
):
    x: kgpy.uncertainty.ArrayLike = 0 * u.deg
    y: kgpy.uncertainty.ArrayLike = 0 * u.deg

    @property
    def cartesian(self: SphericalT) -> kgpy.vectors.Cartesian3D:
        transform = kgpy.transforms.TransformList([
            kgpy.transforms.RotationY(-self.x),
            kgpy.transforms.RotationX(self.y),
        ])
        return transform(kgpy.vectors.Cartesian3D.z_hat(), translate=False)


@dataclasses.dataclass(eq=False)
class StokesVector(
    kgpy.vectors.AbstractVector,
):
    import astropy.units

    i: kgpy.uncertainty.ArrayLike = 1 * astropy.units.dimensionless_unscaled
    q: kgpy.uncertainty.ArrayLike = 0 * astropy.units.dimensionless_unscaled
    u: kgpy.uncertainty.ArrayLike = 0 * astropy.units.dimensionless_unscaled
    v: kgpy.uncertainty.ArrayLike = 0 * astropy.units.dimensionless_unscaled


@dataclasses.dataclass(eq=False)
class FieldVector(kgpy.vectors.AbstractVector):
    field_x: kgpy.uncertainty.ArrayLike = 0 * u.deg
    field_y: kgpy.uncertainty.ArrayLike = 0 * u.deg

    @property
    def field_xy(self) -> kgpy.vectors.Cartesian2D:
        return kgpy.vectors.Cartesian2D(self.field_x, self.field_y)


@dataclasses.dataclass(eq=False)
class PupilVector(kgpy.vectors.AbstractVector):
    pupil_x: kgpy.uncertainty.ArrayLike = 0 * u.mm
    pupil_y: kgpy.uncertainty.ArrayLike = 0 * u.mm

    @property
    def pupil_xy(self) -> kgpy.vectors.Cartesian2D:
        return kgpy.vectors.Cartesian2D(self.pupil_x, self.pupil_y)


@dataclasses.dataclass(eq=False)
class PositionVector(kgpy.vectors.AbstractVector):
    position_x: kgpy.uncertainty.ArrayLike = 0 * u.mm
    position_y: kgpy.uncertainty.ArrayLike = 0 * u.mm

    @property
    def position(self) -> 'PositionVector':
        return PositionVector(self.position_x, self.position_y)

    @property
    def position_xy(self):
        return kgpy.vectors.Cartesian2D(self.position_x, self.position_y)


@dataclasses.dataclass(eq=False)
class SpectralFieldVector(
    FieldVector,
    kgpy.vectors.SpectralVector,
):

    def to_matrix(self):
        from . import matrix
        return matrix.SpectralFieldMatrix(wavelength=self.wavelength, field_x=self.field_x, field_y=self.field_y)

    pass


@dataclasses.dataclass(eq=False)
class SpectralPositionVector(
    PositionVector,
    kgpy.vectors.SpectralVector,
):
    pass


@dataclasses.dataclass(eq=False)
class ObjectVector(
    PupilVector,
    FieldVector,
    kgpy.vectors.DopplerVector,
):
    pass


@dataclasses.dataclass(eq=False)
class SpotVector(
    PositionVector,
    FieldVector,
    kgpy.vectors.DopplerVector,
):
    pass


@dataclasses.dataclass(eq=False)
class InputAngleVector(
    kgpy.vectors.SpectralVector,
):
    angle_input_x: kgpy.uncertainty.ArrayLike = 0 * u.deg
    angle_input_y: kgpy.uncertainty.ArrayLike = 0 * u.deg


@dataclasses.dataclass(eq=False)
class InputOutputAngleVector(
    InputAngleVector,
):
    angle_output_x: kgpy.uncertainty.ArrayLike = 0 * u.deg
    angle_output_y: kgpy.uncertainty.ArrayLike = 0 * u.deg