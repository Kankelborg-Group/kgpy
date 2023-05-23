import typing as typ
import dataclasses
import astropy.units as u
import kgpy.uncertainty
import kgpy.optics
import kgpy.function

__all__ = ['angular_radius_max']

# https://en.wikipedia.org/wiki/Angular_diameter#Use_in_astronomy
angular_radius_max = (32 * u.arcmin + 32 * u.arcsec) / 2  # type: u.Quantity
"""maximum angular radius of the solar disk"""

RadianceInputT = typ.TypeVar('RadianceInputT', bound=kgpy.optics.vectors.TemporalSpectralFieldVector)
RadianceOutputT = typ.TypeVar('RadianceOutputT', bound=kgpy.uncertainty.ArrayLike)

SpectralRadianceInputT = typ.TypeVar('SpectralRadianceInputT', bound=kgpy.optics.vectors.TemporalOffsetSpectralFieldVector)
SpectralRadianceOutputT = typ.TypeVar('SpectralRadianceOutputT', bound=kgpy.uncertainty.ArrayLike)


@dataclasses.dataclass(eq=False)
class Radiance(
    kgpy.function.Array[RadianceInputT, RadianceOutputT]
):
    pass


@dataclasses.dataclass(eq=False)
class SpectralRadiance(
    kgpy.function.Array[SpectralRadianceInputT, SpectralRadianceOutputT]
):
    pass
