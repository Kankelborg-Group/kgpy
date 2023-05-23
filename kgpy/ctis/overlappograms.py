from __future__ import annotations
from typing import TypeVar
import dataclasses
import kgpy.uncertainty
import kgpy.function
from . import vectors

__all__ = [
    'Overlappogram',
]

InputT = TypeVar('InputT', bound=vectors.DispersionOffsetSpectralPositionVector)
OutputT = TypeVar('OutputT', bound=kgpy.uncertainty.ArrayLike)


@dataclasses.dataclass
class Overlappogram(
    kgpy.function.Array[InputT, OutputT],
):
    pass
