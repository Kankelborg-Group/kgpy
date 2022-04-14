import typing as typ
import abc
import dataclasses
import kgpy.labeled
import kgpy.uncertainty
import kgpy.function
import kgpy.optics
from .. import vectors
from .. import instruments

__all__ = [
    'AbstractInversion'
]

AbstractInversionT = typ.TypeVar('AbstractInversionT', bound='AbstractInversion')


@dataclasses.dataclass
class AbstractInversion(
    abc.ABC,
):

    @abc.abstractmethod
    def __call__(
            self: AbstractInversionT,
            instrument: instruments.AbstractInstrument,
            image: kgpy.function.Array[vectors.PixelVector, kgpy.labeled.Array],

    ) -> kgpy.function.Array[kgpy.optics.vectors.SpectralFieldVector, kgpy.labeled.Array]:

        pass
