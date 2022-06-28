import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
import kgpy.uncertainty
import kgpy.vectors
import kgpy.matrix
from . import vectors

SpectralFieldMatrixT = typ.TypeVar('SpectralFieldMatrixT', bound='SpectralFieldMatrix')
SpectralPositionMatrixT = typ.TypeVar('SpectralPositionMatrixT', bound='SpectralPositionMatrix')


@dataclasses.dataclass(eq=False)
class SpectralPositionMatrix(
    vectors.SpectralPositionVector,
    kgpy.matrix.AbstractMatrix,
):
    @classmethod
    def identity(cls: typ.Type[SpectralPositionMatrixT]) -> SpectralPositionMatrixT:
        raise NotImplementedError

    def to_vector(self: SpectralPositionMatrixT) -> kgpy.vectors.AbstractVector:
        return vectors.SpectralPositionVector(
            position_x=self.position_x,
            position_y=self.position_y,
            wavelength=self.wavelength,
        )


@dataclasses.dataclass(eq=False)
class SpectralFieldMatrix(
    vectors.SpectralFieldVector,
    kgpy.matrix.AbstractMatrix,
):
    @classmethod
    def rotation_spatial(cls, angle: kgpy.uncertainty.ArrayLike) -> SpectralFieldMatrixT:
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return cls(
            wavelength=vectors.SpectralFieldVector(wavelength=1, field_x=0*u.AA/u.arcsec, field_y=0 * u.AA/u.arcsec),
            field_x=vectors.SpectralFieldVector(wavelength=0 * u.arcsec/u.AA, field_x=cos_a, field_y=-sin_a),
            field_y=vectors.SpectralFieldVector(wavelength=0 * u.arcsec/u.AA, field_x=sin_a, field_y=cos_a),
        )

    @classmethod
    def identity(cls: typ.Type[SpectralFieldMatrixT]) -> SpectralFieldMatrixT:
        raise NotImplementedError

    def to_vector(self: SpectralFieldMatrixT) -> kgpy.vectors.AbstractVector:
        return vectors.SpectralFieldVector(
            field_x=self.field_x,
            field_y=self.field_y,
            wavelength=self.wavelength,
        )
