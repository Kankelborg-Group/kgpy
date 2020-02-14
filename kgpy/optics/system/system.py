import dataclasses
import pathlib
import pickle
import numpy as np
import typing as typ
from scipy.spatial.transform import Rotation
import astropy.units as u
import astropy.visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from . import mixin
from . import Surface, Fields, Wavelengths, surface, Rays

__all__ = ['System']

SurfacesT = typ.TypeVar('SurfacesT', bound=typ.Iterable[Surface])


@dataclasses.dataclass
class System(mixin.Named, typ.Generic[SurfacesT]):

    surfaces: SurfacesT = dataclasses.field(default_factory=lambda: [])
    fields: Fields = dataclasses.field(default_factory=lambda: Fields())
    wavelengths: Wavelengths = dataclasses.field(default_factory=lambda: Wavelengths())
    entrance_pupil_radius: u.Quantity = 0 * u.m
    # stop_surface_index: typ.Union[int, np.ndarray] = 1
    num_pupil_rays: typ.Tuple[int, int] = (7, 7)
    num_field_rays: typ.Tuple[int, int] = (7, 7)
    raytrace_path: pathlib.Path = dataclasses.field(default_factory=lambda: pathlib.Path())

    @property
    def config_broadcast(self):
        all_surface_battrs = None
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

    @property
    def stop_surface_index(self):
        for i, s in enumerate(self.surfaces):
            if s.is_stop:
                return i + 1

    @property
    def raytrace(self):

        if self.raytrace_path.exists():
            with open(str(self.raytrace_path), 'rb') as f:
                return pickle.load(f)

        else:
            from kgpy.optics import zemax

            zemax_system, zemax_units = zemax.system.calc_zemax_system(self)
            zemax_system.SaveAs(self.raytrace_path.parent / self.raytrace_path.stem / '.zmx')

            raytrace = zemax.system.rays.trace(zemax_system, zemax_units, self.num_pupil_rays, self.num_field_rays,
                                               surface_indices=None)

            with open(str(self.raytrace_path), 'wb') as f:
                pickle.dump(raytrace, f)

            return raytrace

    def raytrace_to_image(
            self,
            num_pupil: typ.Union[int, typ.Tuple[int, int]] = 5,
            num_field: typ.Union[int, typ.Tuple[int, int]] = 5,
    ):
        from kgpy.optics import zemax

        zemax_system, zemax_units = zemax.system.calc_zemax_system(self)
        zemax_system.SaveAs(self.raytrace_path.parent / self.raytrace_path.stem / '.zmx')

        return zemax.system.rays.trace(zemax_system, zemax_units, num_pupil, num_field)

    @property
    def chief_ray(self):
        return self.raytrace.pupil_mean.field_mean

    def local_to_global(self, local_surface: Surface, x: u.Quantity,
                        configuration_axis: typ.Optional[typ.Union[int, typ.Tuple[int, ...]]] = None) -> u.Quantity:
        """
        Convert from the local coordinates of a particular surface to global coordinates.
        :param local_surface: The local surface and the origin of the coordinate system
        :param x: Array of coordinates, may be of any shape.
        :param configuration_axis: The axis or axes of `x` corresponding to different configurations of the optical
        system.
        If None, the configuration axis is created at the zeroth axis of `x`.
        :return: If configuration axis is not None, a new array is returned the same shape as `x`.
        If None, the returned shape is `self.config_broadcast.shape + x.shape`
        """

        if configuration_axis is None:
            x = np.broadcast_to(x.value, self.config_broadcast.shape + x.shape) << x.unit
            configuration_axis = 0

        if np.ndim(configuration_axis) == 0:
            configuration_axis = [configuration_axis]

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

        x = x.copy()

        x_global = x

        for axis in reversed(configuration_axis):
            x = np.moveaxis(x, axis, 0)

        xsh = x.shape

        x = x.reshape(xsh[:len(configuration_axis)] + (-1, 3))

        for i in range(x.shape[len(configuration_axis)]):
            j = ..., i, slice(None)
            x[j] = rotation.apply(x[j]) << x.unit
            x[j] = x[j] + translation

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

    def distance_between_surfaces(self, surface_1: Surface, surface_2: Surface):

        surface_1_index = self.surfaces.index(surface_1)
        surface_2_index = self.surfaces.index(surface_2)

        s1 = [slice(None)] * self.chief_ray.axis.num_axes
        s2 = [slice(None)] * self.chief_ray.axis.num_axes

        s1[self.chief_ray.axis.surf] = slice(surface_1_index, surface_1_index + 1)
        s2[self.chief_ray.axis.surf] = slice(surface_2_index, surface_2_index + 1)

        x1 = self.chief_ray.position[s1]
        x2 = self.chief_ray.position[s2]

        x1 = self.local_to_global(surface_1, x1, configuration_axis=self.chief_ray.axis.config)
        x2 = self.local_to_global(surface_2, x2, configuration_axis=self.chief_ray.axis.config)

        return np.sqrt(np.sum(np.square(x2 - x1), axis=~0, keepdims=True))

    def plot_3d(self, rays: Rays, delete_vignetted=True, config_index: int = 0):

        with astropy.visualization.quantity_support():

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', proj_type='ortho')

            x, mask = rays.position, rays.mask

            xsl = [slice(None)] * len(x.shape)
            for surf in self.surfaces:
                s = self.surfaces.index(surf)
                xsl[rays.axis.surf] = s
                a = self.local_to_global(surf, x[xsl], configuration_axis=0)
                x[xsl] = a[config_index]

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
            sl = (
            0, 0, slice(None, None, rebin_factor), slice(None, None, rebin_factor), slice(None, None, rebin_factor),
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

                        polys = self.local_to_global(surf, polys, configuration_axis=None)[0]

                        for poly in polys:
                            x_, y_, z_ = poly.T
                            ax.plot(x_, y_, z_, 'k')
                            ax.plot(u.Quantity([x_[-1], x_[0]]), u.Quantity([y_[-1], y_[0]]),
                                    u.Quantity([z_[-1], z_[0]]), 'k')
