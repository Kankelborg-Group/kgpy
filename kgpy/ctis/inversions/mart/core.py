import typing as typ
import abc
import dataclasses
import kgpy.uncertainty
import kgpy.function
import kgpy.optics
from ... import vectors
from .. import abstractions

__all__ = [
    'MART_Inversion'
]

MART_InversionT = typ.TypeVar('MART_InversionT', bound='MART_Inversion')


@dataclasses.dataclass
class MART_Inversion(
    abstractions.AbstractInversion
):

    @abc.abstractmethod
    def __call__(
            self: MART_InversionT,
            image: kgpy.function.Array[vectors.DispersionOffsetSpectralPositionVector, kgpy.uncertainty.ArrayLike]
    ) -> kgpy.function.Array[kgpy.optics.vectors.SpectralFieldVector, kgpy.uncertainty.ArrayLike]:
        pass



