import abc
import dataclasses
import numpy as np
import typing as tp
from scipy.spatial.transform import Rotation
import astropy.units as u

from . import Surface, Fields, Wavelengths, surface

__all__ = ['System']


@dataclasses.dataclass
class System:
    name: str
    surfaces: tp.List[Surface]
    fields: Fields
    wavelengths: Wavelengths
    entrance_pupil_radius: u.Quantity = 0 * u.m
    stop_surface_index: tp.Union[int, np.ndarray] = 1

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

    def local_to_global(self, local_surface: Surface, x: u.Quantity) -> u.Quantity:

        xsh = x.shape[:~0]
        sh = xsh + (1,) * len(self.config_broadcast.shape) + (3,)
        x = np.reshape(x, sh)

        surface_index = self.surfaces.index(local_surface)

        translation = [0, 0, 0] * u.mm
        rotation = Rotation.from_euler('XYZ', [0, 0, 0])  # type: Rotation

        for s, surf in enumerate(self.surfaces[:surface_index + 1]):

            if isinstance(surf, surface.Standard):

                rotation, translation = transform(translation, rotation, surf.decenter_before,
                                                  surf.tilt_before, surf.tilt_first)

                if s < surface_index:
                    rotation, translation = transform(translation, rotation, surf.decenter_after, surf.tilt_after,
                                                      ~surf.tilt_first)

            elif isinstance(surf, surface.CoordinateBreak):

                rotation, translation = transform(translation, rotation, surf.decenter, surf.tilt, surf.tilt_first)

            if s < surface_index:
                translation = translation + rotation.apply([0, 0, surf.thickness.value]) * surf.thickness.unit

        x_global = x + 0*translation
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x_global[i, j] = rotation.apply(x_global[i, j].value) * x_global.unit

        x_global = x_global + translation

        return x_global


def transform(
        current_translation: u.Quantity,
        current_rotation: Rotation,
        next_translation: u.Quantity,
        next_euler_angles: u.Quantity,
        tilt_first: bool,
):

    next_translation = next_translation.copy()
    next_translation[~0] = 0 * u.mm

    if not tilt_first:

        translation = current_translation + current_rotation.apply(next_translation.value) * next_translation.unit
        rotation = current_rotation * Rotation.from_euler('XYZ', next_euler_angles.to(u.rad))

    else:

        rotation = current_rotation * Rotation.from_euler('ZYX', np.flip(next_euler_angles.to(u.rad), axis=~0))
        translation = current_translation + rotation.apply(next_translation.value) * next_translation.unit

    return rotation, translation
