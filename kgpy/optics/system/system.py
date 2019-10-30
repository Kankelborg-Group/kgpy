import dataclasses
import numpy as np
import typing as tp
import nptyping as npt
import astropy.units as u

from . import Surface, Fields, Wavelengths

__all__ = ['System']


@dataclasses.dataclass
class System:

    name: str
    surfaces: tp.Iterable[Surface]
    fields: Fields
    wavelengths: Wavelengths
    entrance_pupil_diameter: u.Quantity = 0 * u.m

    @property
    def config_broadcast(self):

        all_surface_battrs = np.broadcast(*[s.config_broadcast for s in self.surfaces])

        return np.broadcast(
            all_surface_battrs,
            self.fields.config_broadcast,
            self.wavelengths.config_broadcast,
        )
