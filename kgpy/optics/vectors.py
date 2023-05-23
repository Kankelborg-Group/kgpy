from __future__ import annotations
import typing as typ
import abc
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

TimeT = typ.TypeVar('TimeT', bound=kgpy.uncertainty.ArrayLike)
WavelengthT = typ.TypeVar('WavelengthT', bound=kgpy.uncertainty.ArrayLike)
WavelengthBaseT = typ.TypeVar('WavelengthBaseT', bound=kgpy.uncertainty.ArrayLike)
WavelengthOffsetT = typ.TypeVar('WavelengthOffsetT', bound=kgpy.uncertainty.ArrayLike)
WavelengthRestT = typ.TypeVar('WavelengthRestT', bound=kgpy.uncertainty.ArrayLike)
FieldXT = typ.TypeVar('FieldXT', bound=kgpy.uncertainty.ArrayLike)
FieldYT = typ.TypeVar('FieldYT', bound=kgpy.uncertainty.ArrayLike)
PupilXT = typ.TypeVar('PupilXT', bound=kgpy.uncertainty.ArrayLike)
PupilYT = typ.TypeVar('PupilYT', bound=kgpy.uncertainty.ArrayLike)
PositionXT = typ.TypeVar('PositionXT', bound=kgpy.uncertainty.ArrayLike)
PositionYT = typ.TypeVar('PositionYT', bound=kgpy.uncertainty.ArrayLike)
VelocityT = typ.TypeVar('VelocityT', bound=kgpy.uncertainty.ArrayLike)
SphericalT = typ.TypeVar('SphericalT', bound='Spherical')
StokesVectorT = typ.TypeVar('StokesVectorT', bound='StokesVector')
AbstractSpectralVectorT = typ.TypeVar('AbstractSpectralVectorT', bound='AbstractSpectralVector')
SpectralVectorT = typ.TypeVar('SpectralVectorT', bound='SpectralVector')
SpectralReferenceVectorT = typ.TypeVar('SpectralReferenceVectorT', bound='SpectralReferenceVector')
DopplerVectorT = typ.TypeVar('DopplerVectorT', bound='DopplerVector')
OffsetSpectralFieldVectorT = typ.TypeVar('OffsetSpectralFieldVectorT', bound='OffsetSpectralFieldVector')
SpectralFieldVectorT = typ.TypeVar('SpectralFieldVectorT', bound='SpectralFieldVector')
OffsetSpectralPositionVectorT = typ.TypeVar('OffsetSpectralPositionVectorT', bound='OffsetSpectralPositionVector')


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
class AbstractSpectralVector(
    kgpy.vectors.VectorInterface,
    typ.Generic[WavelengthT],
):

    @property
    @abc.abstractmethod
    def wavelength(self: AbstractSpectralVectorT) -> WavelengthT:
        pass

    @property
    def vector_spectral(self: AbstractSpectralVectorT) -> 'SpectralVector':
        return SpectralVector(self.wavelength)


@dataclasses.dataclass(eq=False)
class SpectralVector(
    kgpy.vectors.AbstractVector,
    AbstractSpectralVector[WavelengthT],
):
    wavelength: WavelengthT = 0 * u.nm


@dataclasses.dataclass(eq=False)
class OffsetSpectralVector(
    AbstractSpectralVector,
    kgpy.vectors.AbstractVector,
    typ.Generic[WavelengthBaseT, WavelengthOffsetT],
):
    wavelength_base: WavelengthBaseT = 0 * u.nm
    wavelength_offset: WavelengthOffsetT = 0 * u.nm

    @property
    def wavelength(self: SpectralReferenceVectorT) -> kgpy.uncertainty.ArrayLike:
        return self.wavelength_base + self.wavelength_offset

    @property
    def velocity_los(self):
        return (astropy.constants.c * self.wavelength_offset / self.wavelength).to(u.km / u.s)


@dataclasses.dataclass(eq=False)
class DopplerVector(
    AbstractSpectralVector,
    kgpy.vectors.AbstractVector,
    typ.Generic[WavelengthRestT, VelocityT],
):
    wavelength_rest: WavelengthRestT = 0 * u.nm
    velocity_los: VelocityT = 0 * u.km / u.s

    @property
    def wavelength(self: DopplerVectorT) -> kgpy.uncertainty.ArrayLike:
        return self.wavelength_rest * (1 + self.velocity_los / astropy.constants.c)


@dataclasses.dataclass(eq=False)
class FieldVector(
    kgpy.vectors.AbstractVector,
    typ.Generic[FieldXT, FieldYT],
):
    field_x: FieldXT = 0 * u.deg
    field_y: FieldYT = 0 * u.deg

    @property
    def field_xy(self) -> kgpy.vectors.Cartesian2D[FieldXT, FieldYT]:
        return kgpy.vectors.Cartesian2D(self.field_x, self.field_y)


@dataclasses.dataclass(eq=False)
class PupilVector(
    kgpy.vectors.AbstractVector,
    typ.Generic[PupilXT, PupilYT],
):
    pupil_x: PupilXT = 0 * u.mm
    pupil_y: PupilYT = 0 * u.mm

    @property
    def pupil_xy(self) -> kgpy.vectors.Cartesian2D[PupilXT, PupilYT]:
        return kgpy.vectors.Cartesian2D(self.pupil_x, self.pupil_y)


@dataclasses.dataclass(eq=False)
class PositionVector(
    kgpy.vectors.AbstractVector,
    typ.Generic[PositionXT, PositionYT]
):
    position_x: PositionXT = 0 * u.mm
    position_y: PositionYT = 0 * u.mm

    @property
    def position(self) -> 'PositionVector':
        return PositionVector(self.position_x, self.position_y)

    @property
    def position_xy(self) -> kgpy.vectors.Cartesian2D[PositionXT, PositionYT]:
        return kgpy.vectors.Cartesian2D(self.position_x, self.position_y)


@dataclasses.dataclass(eq=False)
class SpectralFieldVector(
    typ.Generic[WavelengthT, FieldXT, FieldYT],
    FieldVector[FieldXT, FieldYT],
    SpectralVector[WavelengthT],
):
    @property
    def type_matrix(self):
        from . import matrix
        return matrix.SpectralFieldMatrix


@dataclasses.dataclass(eq=False)
class OffsetSpectralFieldVector(
    typ.Generic[WavelengthBaseT, WavelengthOffsetT, FieldXT, FieldYT],
    FieldVector[FieldXT, FieldYT],
    OffsetSpectralVector[WavelengthBaseT, WavelengthOffsetT],
):
    @property
    def vector_spectral_field(self: OffsetSpectralFieldVectorT) -> SpectralFieldVector:
        return SpectralFieldVector(
            wavelength=self.wavelength,
            field_x=self.field_x,
            field_y=self.field_y,
        )


@dataclasses.dataclass(eq=False)
class TemporalSpectralFieldVector(
    typ.Generic[TimeT, WavelengthT, FieldXT, FieldYT],
    SpectralFieldVector[WavelengthT, FieldXT, FieldYT],
    kgpy.vectors.TemporalVector[TimeT],
):
    pass


@dataclasses.dataclass
class TemporalOffsetSpectralFieldVector(
    typ.Generic[TimeT, WavelengthBaseT, WavelengthOffsetT, FieldXT, FieldYT],
    OffsetSpectralFieldVector[WavelengthBaseT, WavelengthOffsetT, FieldXT, FieldYT],
    kgpy.vectors.TemporalVector[TimeT],
):
    pass


@dataclasses.dataclass(eq=False)
class SpectralPositionVector(
    typ.Generic[WavelengthT, PositionXT, PositionYT],
    PositionVector[PositionXT, PositionYT],
    SpectralVector[WavelengthT],
):
    @property
    def type_matrix(self):
        from . import matrix
        return matrix.SpectralPositionMatrix

    @property
    def vector_spectral_position(self: OffsetSpectralPositionVectorT) -> SpectralPositionVector:
        return SpectralPositionVector(
            wavelength=self.wavelength,
            position_x=self.position_x,
            position_y=self.position_y,
        )


@dataclasses.dataclass(eq=False)
class OffsetSpectralPositionVector(
    PositionVector[PositionXT, PositionYT],
    OffsetSpectralVector[WavelengthBaseT, WavelengthOffsetT],
):
    @property
    def vector_spectral_position(self: OffsetSpectralPositionVectorT) -> SpectralPositionVector:
        return SpectralPositionVector(
            wavelength=self.wavelength,
            position_x=self.position_x,
            position_y=self.position_y,
        )


@dataclasses.dataclass(eq=False)
class TemporalSpectralPositionVector(
    typ.Generic[TimeT, WavelengthT, PositionXT, PositionYT],
    SpectralPositionVector[WavelengthT, PositionXT, PositionYT],
    kgpy.vectors.TemporalVector[TimeT],
):
    pass


@dataclasses.dataclass(eq=False)
class TemporalOffsetSpectralPositionVector(
    typ.Generic[TimeT, WavelengthBaseT, WavelengthOffsetT, PositionXT, PositionYT],
    OffsetSpectralPositionVector[WavelengthBaseT, WavelengthOffsetT, PositionXT, PositionYT],
    kgpy.vectors.TemporalVector[TimeT],
):
    pass


@dataclasses.dataclass(eq=False)
class ObjectVector(
    typ.Generic[WavelengthBaseT, WavelengthOffsetT, FieldXT, FieldYT, PupilXT, PupilYT],
    PupilVector[PupilXT, PupilYT],
    OffsetSpectralFieldVector[WavelengthBaseT, WavelengthOffsetT, FieldXT, FieldYT],
):
    pass


@dataclasses.dataclass(eq=False)
class SpotVector(
    typ.Generic[WavelengthBaseT, WavelengthOffsetT, FieldXT, FieldYT, PositionXT, PositionYT],
    PositionVector[PositionXT, PositionYT],
    FieldVector[FieldXT, FieldYT],
    OffsetSpectralVector[WavelengthBaseT, WavelengthOffsetT],
):
    pass


@dataclasses.dataclass(eq=False)
class InputAngleVector(
    SpectralVector,
):
    angle_input_x: kgpy.uncertainty.ArrayLike = 0 * u.deg
    angle_input_y: kgpy.uncertainty.ArrayLike = 0 * u.deg


@dataclasses.dataclass(eq=False)
class InputOutputAngleVector(
    InputAngleVector,
):
    angle_output_x: kgpy.uncertainty.ArrayLike = 0 * u.deg
    angle_output_y: kgpy.uncertainty.ArrayLike = 0 * u.deg