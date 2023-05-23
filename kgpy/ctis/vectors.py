import typing as typ
import dataclasses
import astropy.units as u
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vectors
import kgpy.optics.vectors

__all__ = [
    'DispersionVector',
    'DispersionOffsetSpectralPositionVector',
]

TimeT = typ.TypeVar('TimeT', bound=kgpy.labeled.ArrayLike)
SpectralOrderT = typ.TypeVar('SpectralOrderT', bound=kgpy.labeled.ArrayLike)
AngleDispersionT = typ.TypeVar('AngleDispersionT', bound=kgpy.uncertainty.ArrayLike)
WavelengthBaseT = typ.TypeVar('WavelengthBaseT', bound=kgpy.uncertainty.ArrayLike)
WavelengthOffsetT = typ.TypeVar('WavelengthOffsetT', bound=kgpy.uncertainty.ArrayLike)
PositionXT = typ.TypeVar('PositionXT', bound=kgpy.uncertainty.ArrayLike)
PositionYT = typ.TypeVar('PositionYT', bound=kgpy.uncertainty.ArrayLike)


@dataclasses.dataclass(eq=False)
class DispersionVector(
    kgpy.vectors.AbstractVector,
    typ.Generic[SpectralOrderT, AngleDispersionT]
):
    spectral_order: SpectralOrderT = 0
    angle_dispersion: AngleDispersionT = 0 * u.deg


@dataclasses.dataclass(eq=False)
class DispersionOffsetSpectralPositionVector(
    typ.Generic[SpectralOrderT, AngleDispersionT, WavelengthBaseT, WavelengthOffsetT, PositionXT, PositionYT],
    kgpy.optics.vectors.OffsetSpectralPositionVector[WavelengthBaseT, WavelengthOffsetT, PositionXT, PositionYT],
    DispersionVector[SpectralOrderT, AngleDispersionT],
):
    pass


@dataclasses.dataclass(eq=False)
class TemporalDispersionOffsetSpectralPositionVector(
    typ.Generic[TimeT, SpectralOrderT, AngleDispersionT, WavelengthBaseT, WavelengthOffsetT, PositionXT, PositionYT],
    DispersionOffsetSpectralPositionVector[SpectralOrderT, AngleDispersionT, WavelengthBaseT, WavelengthOffsetT, PositionXT, PositionYT],
    kgpy.vectors.TemporalVector[TimeT],
):
    pass