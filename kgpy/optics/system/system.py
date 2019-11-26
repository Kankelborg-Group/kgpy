import dataclasses
import numpy as np
import typing as tp
from scipy.spatial.transform import Rotation
import astropy.units as u
import matplotlib.pyplot as plt

from . import Surface, Fields, Wavelengths, surface, Rays

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

                rotation, translation = self._transform(translation, rotation, surf.decenter_before,
                                                        surf.tilt_before, surf.tilt_first)

                if s < surface_index:
                    rotation, translation = self._transform(translation, rotation, surf.decenter_after, surf.tilt_after,
                                                            ~surf.tilt_first)

            elif isinstance(surf, surface.CoordinateBreak):

                rotation, translation = self._transform(translation, rotation, surf.decenter, surf.tilt,
                                                        surf.tilt_first)

            if s < surface_index:
                translation = translation + rotation.apply([0, 0, surf.thickness.value]) * surf.thickness.unit

        x_global = x + 0 * translation
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x_global[i, j] = rotation.apply(x_global[i, j].value) * x_global.unit

        x_global = x_global + translation

        return x_global

    @staticmethod
    def _transform(
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

    def plot_3d(self, rays: Rays, delete_vignetted=True):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', proj_type='ortho')

        x, mask = rays.position, rays.mask

        xsl = [slice(None)] * len(x.shape)
        for surf in self.surfaces:
            s = self.surfaces.index(surf)
            xsl[rays.axis.surf] = s
            x[xsl] = self.local_to_global(surf, x[xsl])

        new_order = [
            rays.axis.config,
            rays.axis.wavl,
            rays.axis.field_x,
            rays.axis.field_y,
            rays.axis.pupil_x,
            rays.axis.pupil_y,
            rays.axis.surf,
            ~0,
        ]

        new_surf_axis = ~list(reversed(new_order)).index(rays.axis.surf)

        x = np.transpose(x, new_order)
        mask = np.transpose(mask, new_order)

        rebin_factor = 1
        sl = (0, 0, slice(None, None, rebin_factor), slice(None, None, rebin_factor), slice(None, None, rebin_factor),
              slice(None, None, rebin_factor))
        x = x[sl]
        mask = mask[sl]

        sh = (-1, x.shape[new_surf_axis])
        x = np.reshape(x, sh + (3,))
        mask = np.reshape(mask, sh)

        for r, m in zip(x, mask):
            if delete_vignetted:
                if m[~0]:
                    ax.plot(*r.T)
            else:
                ax.plot(*r[m].T)

        for surf in self.surfaces:
            if isinstance(surf, surface.Standard):
                if surf.aperture is not None:

                    psh = list(surf.aperture.points.shape)
                    psh[~0] = 3
                    polys = np.zeros(psh) * u.mm
                    polys[..., 0:2] = surf.aperture.points

                    polys = self.local_to_global(surf, polys)

                    for poly in polys:
                        x_, y_, z_ = poly[:, 0].T
                        ax.plot(x_, y_, z_, 'k')
                        ax.plot(u.Quantity([x_[-1], x_[0]]), u.Quantity([y_[-1], y_[0]]),
                                u.Quantity([z_[-1], z_[0]]), 'k')
