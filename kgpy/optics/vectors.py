import typing as typ
import dataclasses
import astropy.units as u
import astropy.constants
import kgpy.uncertainty
import kgpy.vectors

__all__ = [
    'StokesVector',
    'SpectralVector',
    'SpectralFieldVector',
    'ObjectVector',
    'ImageVector',
]

StokesVectorT = typ.TypeVar('StokesVectorT', bound='StokesVector')
SpectralVectorT = typ.TypeVar('SpectralVectorT', bound='SpectralVector')


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
    velocity_los: kgpy.uncertainty.ArrayLike = 0 * u.km / u.s

    @property
    def wavelength_doppler(self: SpectralVectorT) -> kgpy.uncertainty.ArrayLike:
        return self.wavelength * (1 + self.velocity_los / astropy.constants.c)


@dataclasses.dataclass(eq=False)
class FieldComponents:
    field: kgpy.vectors.Cartesian2D = dataclasses.field(default_factory=kgpy.vectors.Cartesian2D)


@dataclasses.dataclass(eq=False)
class PupilComponents:
    pupil: kgpy.vectors.Cartesian2D = dataclasses.field(default_factory=lambda: kgpy.vectors.Cartesian2D() * u.mm)


@dataclasses.dataclass(eq=False)
class PositionComponents:
    position: kgpy.vectors.Cartesian2D = dataclasses.field(default_factory=lambda: kgpy.vectors.Cartesian2D() * u.mm)


@dataclasses.dataclass(eq=False)
class SpectralFieldVector(
    SpectralVector,
    FieldComponents,
):
    pass


@dataclasses.dataclass(eq=False)
class ObjectVector(
    SpectralVector,
    PupilComponents,
    FieldComponents,
):
    pass


@dataclasses.dataclass(eq=False)
class ImageVector(
    SpectralVector,
    PositionComponents,
    FieldComponents,
):
    pass
