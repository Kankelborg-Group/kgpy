import typing as typ
import dataclasses
import pathlib
import astropy.units as u
import kgpy.chianti
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vectors
import kgpy.function

__all__ = ['angular_radius_max']

# https://en.wikipedia.org/wiki/Angular_diameter#Use_in_astronomy
angular_radius_max = (32 * u.arcmin + 32 * u.arcsec) / 2  # type: u.Quantity
"""maximum angular radius of the solar disk"""

TimeT = typ.TypeVar('TimeT', bound=kgpy.uncertainty.ArrayLike)
WavelengthT = typ.TypeVar('WavelengthT', bound=kgpy.uncertainty.ArrayLike)
WavelengthReferenceT = typ.TypeVar('WavelengthReferenceT', bound=kgpy.uncertainty.ArrayLike)
WavelengthRelativeT = typ.TypeVar('WavelengthRelativeT', bound=kgpy.uncertainty.ArrayLike)
HelioprojectiveXT = typ.TypeVar('HelioprojectiveXT', bound=kgpy.uncertainty.ArrayLike)
HelioprojectiveYT = typ.TypeVar('HelioprojectiveYT', bound=kgpy.uncertainty.ArrayLike)


@dataclasses.dataclass(eq=False)
class HelioprojectiveCartesianVector(
    kgpy.vectors.AbstractVector,
    typ.Generic[HelioprojectiveXT, HelioprojectiveYT],
):
    helioprojective_x: HelioprojectiveXT = 0 * u.arcsec
    helioprojective_y: HelioprojectiveYT = 0 * u.arcsec


@dataclasses.dataclass(eq=False)
class TemporalSpectralHelioprojectiveCartesianVector(
    typ.Generic[TimeT, WavelengthT, HelioprojectiveXT, HelioprojectiveYT],
    HelioprojectiveCartesianVector[HelioprojectiveXT, HelioprojectiveYT],
    kgpy.vectors.SpectralVector[WavelengthT],
    kgpy.vectors.TemporalVector[TimeT],
):
    pass


@dataclasses.dataclass(eq=False)
class RadiantIntensity(
    kgpy.function.Array[
        TemporalSpectralHelioprojectiveCartesianVector[
            kgpy.uncertainty.ArrayLike,
            kgpy.uncertainty.ArrayLike,
            kgpy.labeled.WorldCoordinateSpace[kgpy.vectors.Cartesian2D],
            kgpy.labeled.WorldCoordinateSpace[kgpy.vectors.Cartesian2D],
        ],
        kgpy.uncertainty.ArrayLike,
    ]
):
    pass
