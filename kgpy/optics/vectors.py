import typing as typ
import dataclasses
import astropy.units as u
import astropy.constants
import kgpy.uncertainty
import kgpy.vectors
import kgpy.transforms

__all__ = [
    'StokesVector',
    'SpectralVector',
    'SpectralFieldVector',
    'ObjectVector',
    'SpotVector',
]

SphericalT = typ.TypeVar('SphericalT', bound='Spherical')
StokesVectorT = typ.TypeVar('StokesVectorT', bound='StokesVector')
SpectralVectorT = typ.TypeVar('SpectralVectorT', bound='SpectralVector')


@dataclasses.dataclass(eq=False)
class Spherical(
    kgpy.vectors.AbstractVector,
):
    x: kgpy.uncertainty.ArrayLike = 0 * u.deg
    y: kgpy.uncertainty.ArrayLike= 0 * u.deg

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
class SpectralVector(kgpy.vectors.AbstractVector):
    wavelength: kgpy.uncertainty.ArrayLike = 0 * u.nm


@dataclasses.dataclass(eq=False)
class DopplerVector(kgpy.vectors.AbstractVector):
    wavelength_rest: kgpy.uncertainty.ArrayLike = 0 * u.nm
    velocity_los: kgpy.uncertainty.ArrayLike = 0 * u.km / u.s

    @property
    def wavelength(self: SpectralVectorT) -> kgpy.uncertainty.ArrayLike:
        return self.wavelength_rest * (1 + self.velocity_los / astropy.constants.c)


@dataclasses.dataclass(eq=False)
class FieldVector:
    field_x: kgpy.uncertainty.ArrayLike = 0 * u.deg
    field_y: kgpy.uncertainty.ArrayLike = 0 * u.deg

    @property
    def field_xy(self) -> kgpy.vectors.Cartesian2D:
        return kgpy.vectors.Cartesian2D(self.field_x, self.field_y)


@dataclasses.dataclass(eq=False)
class PupilVector:
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
    def position_xy(self):
        return kgpy.vectors.Cartesian2D(self.position_x, self.position_y)


@dataclasses.dataclass(eq=False)
class SpectralFieldVector(
    SpectralVector,
    FieldVector,
):
    pass


@dataclasses.dataclass(eq=False)
class SpectralPositionVector(
    SpectralVector,
    PositionVector,
):
    pass


@dataclasses.dataclass(eq=False)
class ObjectVector(
    DopplerVector,
    PupilVector,
    FieldVector,
):
    pass


@dataclasses.dataclass(eq=False)
class SpotVector(
    DopplerVector,
    PositionVector,
    FieldVector,
):
    pass


@dataclasses.dataclass(eq=False)
class InputAngleVector(
    SpectralVector,
):
    angle_input: Spherical = dataclasses.field(default_factory=Spherical)


@dataclasses.dataclass(eq=False)
class InputOutputAngleVector(
    InputAngleVector,
):
    angle_output: Spherical = dataclasses.field(default_factory=Spherical)