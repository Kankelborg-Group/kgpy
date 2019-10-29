import dataclasses
import numpy as np
import typing as tp
import nptyping as npt

from . import Surface, Fields, Wavelengths

__all__ = ['System']


@dataclasses.dataclass
class System:

    name: str
    surfaces: tp.Iterable[Surface]
    fields: Fields
    wavelengths: Wavelengths

    @property
    def broadcasted_attrs(self):

        all_surface_battrs = np.broadcast(*[s.broadcasted_attrs for s in self.surfaces])

        return np.broadcast(
            all_surface_battrs,
            self.fields,
            self.wavelengths,
        )
