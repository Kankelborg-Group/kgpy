import abc
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
    surfaces: tp.List[Surface]
    fields: Fields
    wavelengths: Wavelengths
    entrance_pupil_radius: u.Quantity = 0 * u.m
    stop_surface_index: tp.Union[int, npt.Array[int]] = 1

    @property
    def config_broadcast(self):
        all_surface_battrs = 0
        for s in self.surfaces:
            all_surface_battrs = np.broadcast(all_surface_battrs, s.config_broadcast)
            all_surface_battrs = np.empty(all_surface_battrs.shape)

        return np.broadcast(
            all_surface_battrs,
            self.fields.config_broadcast,
            self.wavelengths.config_broadcast,
            self.entrance_pupil_radius,
            self.stop_surface_index,
        )
