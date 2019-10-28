import dataclasses
import nptyping as npt

from . import Surface, Fields, Wavelengths

__all__ = ['System']


@dataclasses.dataclass
class System:

    name: str
    surfaces: npt.Array[Surface]
    fields: Fields
    wavelengths: Wavelengths
