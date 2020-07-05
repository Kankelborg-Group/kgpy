import dataclasses
import pathlib
import pickle
import numpy as np
import typing as typ
import scipy.spatial.transform
import astropy.units as u
import astropy.visualization
import matplotlib.pyplot as plt
import matplotlib.colors
import kgpy.mixin
import kgpy.vector
import kgpy.linspace
from kgpy.vector import x, y, z, ix, iy, iz, xy
import kgpy.optimization.minimization
import kgpy.optimization.root_finding
from .. import ZemaxCompatible, Rays, material, surface, aperture

__all__ = ['System']

SurfacesT = typ.TypeVar('SurfacesT', bound=typ.Union[typ.Iterable[surface.Surface], ZemaxCompatible])


@dataclasses.dataclass
class System(ZemaxCompatible, kgpy.mixin.Broadcastable, kgpy.mixin.Named, typ.Generic[SurfacesT]):
    object_surface: surface.ObjectSurface = dataclasses.field(default_factory=lambda: surface.ObjectSurface())
    surfaces: SurfacesT = dataclasses.field(default_factory=lambda: [])
    stop_surface: typ.Optional[surface.Standard] = None
    wavelengths: typ.Optional[u.Quantity] = None
    pupil_samples: typ.Union[int, typ.Tuple[int, int]] = 3
    field_min: typ.Optional[u.Quantity] = None
    field_max: typ.Optional[u.Quantity] = None
    field_samples: typ.Union[int, typ.Tuple[int, int]] = 3
    field_mask_func: typ.Optional[typ.Callable[[u.Quantity, u.Quantity], np.ndarray]] = None

    @property
    def standard_surfaces(self) -> typ.Iterator[surface.Standard]:
        for s in self.surfaces:
            if isinstance(s, surface.Standard):
                yield s

    @property
    def aperture_surfaces(self) -> typ.Iterator[surface.Standard]:
        for s in self.standard_surfaces:
            if s.aperture is not None:
                if s.aperture.is_active:
                    yield s

    @staticmethod
    def _normalize_2d_samples(samples: typ.Union[int, typ.Tuple[int, int]]) -> typ.Tuple[int, int]:
        if isinstance(samples, int):
            samples = samples, samples
        return samples

    @property
    def pupil_samples_normalized(self) -> typ.Tuple[int, int]:
        return self._normalize_2d_samples(self.pupil_samples)

    @property
    def field_samples_normalized(self) -> typ.Tuple[int, int]:
        return self._normalize_2d_samples(self.field_samples)

    @property
    def image_surface(self) -> surface.Surface:
        s = list(self)
        return s[~0]

    def to_zemax(self) -> 'System':
        from kgpy.optics import zemax
        return zemax.System(
            name=self.name,
            surfaces=self.surfaces.to_zemax(),
        )

    @property
    def config_broadcast(self):
        all_surface_battrs = None
        for s in self.surfaces:
            all_surface_battrs = np.broadcast(all_surface_battrs, s.config_broadcast)
            all_surface_battrs = np.broadcast_to(np.array(1), all_surface_battrs.shape)

        return all_surface_battrs

    @property
    def field_x(self) -> u.Quantity:
        return kgpy.midspace(self.field_min[x], self.field_max[x], self.field_samples_normalized[ix], axis=~0)

    @property
    def field_y(self) -> u.Quantity:
        return kgpy.midspace(self.field_min[y], self.field_max[y], self.field_samples_normalized[iy], axis=~0)

    @property
    def pupil_x(self) -> u.Quantity:
        aper = self.stop_surface.aperture
        return kgpy.midspace(aper.min[x], aper.max[x], self.pupil_samples_normalized[ix], axis=~0)

    @property
    def pupil_y(self) -> u.Quantity:
        aper = self.stop_surface.aperture
        return kgpy.midspace(aper.min[y], aper.max[y], self.pupil_samples_normalized[iy], axis=~0)

    @property
    def input_rays(self):

        x_hat = np.array([1, 0])
        y_hat = np.array([0, 1])

        if np.isinf(self.object_surface.thickness).all():

            position_guess = kgpy.vector.from_components(use_z=False) << u.mm

            step_size = 1 * u.nm
            step = step_size * x_hat + step_size * y_hat

            for surf in self.aperture_surfaces:
                aper = surf.aperture
                amin, amax = aper.min, aper.max
                px = kgpy.midspace(amin[x], amax[x], self.pupil_samples_normalized[ix], axis=~0)
                py = kgpy.midspace(amin[y], amax[y], self.pupil_samples_normalized[iy], axis=~0)
                target_position = kgpy.vector.from_components(np.expand_dims(px, ~0), py)

                def position_error(pos: u.Quantity) -> u.Quantity:
                    position = kgpy.vector.to_3d(pos)
                    rays = Rays.from_field_angles(
                        wavelength_grid=self.wavelengths,
                        position=position,
                        field_grid_x=self.field_x,
                        field_grid_y=self.field_y,
                        field_mask_func=self.field_mask_func,
                        pupil_grid_x=self.pupil_x,
                        pupil_grid_y=self.pupil_y,
                    )
                    rays = self.raytrace_subsystem(rays, final_surface=surf)
                    return (rays.position - target_position)[xy]

                position_guess = kgpy.optimization.root_finding.secant(
                    func=position_error,
                    root_guess=position_guess,
                    step_size=step,
                    max_abs_error=1 * u.nm,
                    max_iterations=100,
                )

                if surf is self.stop_surface:
                    break

            return Rays.from_field_angles(
                wavelength_grid=self.wavelengths,
                position=kgpy.vector.to_3d(position_guess),
                field_grid_x=self.field_x,
                field_grid_y=self.field_y,
                field_mask_func=self.field_mask_func,
                pupil_grid_x=self.pupil_x,
                pupil_grid_y=self.pupil_y,
            )

        else:
            raise NotImplementedError

    def raytrace_subsystem(
            self,
            rays: Rays,
            start_surface: typ.Optional[surface.Surface] = None,
            final_surface: typ.Optional[surface.Surface] = None,
    ) -> Rays:

        surfaces = list(self)

        start_surface_index = 0
        final_surface_index = ~0

        if start_surface is None:
            start_surface = surfaces[start_surface_index]

        if final_surface is None:
            final_surface = surfaces[final_surface_index]

        for s, surf in enumerate(surfaces):
            if surf is start_surface:
                start_surface_index = s
            if surf is final_surface:
                final_surface_index = s
                break

        for s in range(start_surface_index, final_surface_index + 1):
            surf = surfaces[s]
            if s == start_surface_index:
                rays = surf.propagate_rays(rays, is_first_surface=True)
            elif s == final_surface_index:
                rays = surf.propagate_rays(rays, is_final_surface=True)
            else:
                rays = surf.propagate_rays(rays)

        return rays

    @property
    def image_rays(self) -> Rays:
        return self.raytrace_subsystem(self.input_rays)

    @property
    def all_rays(self) -> typ.List[Rays]:

        rays = [self.input_rays]

        old_surf = self.object_surface

        for surf in self.surfaces:
            rays.append(self.raytrace_subsystem(rays[~0], old_surf, surf))
            old_surf = surf

        return rays

    def psf(
            self,
            bins: typ.Union[int, typ.Tuple[int, int]] = 10,
            limits: typ.Optional[typ.Tuple[typ.Tuple[int, int], typ.Tuple[int, int]]] = None,
            use_vignetted: bool = False,
            relative_to_centroid: typ.Tuple[bool, bool] = (False, False),
    ) -> typ.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.image_rays.pupil_hist2d(
            bins=bins,
            limits=limits,
            use_vignetted=use_vignetted,
            relative_to_centroid=relative_to_centroid,
        )

    def plot_footprint(
            self,
            surf: typ.Optional[surface.Standard] = None,
            color_axis: int = Rays.axis.wavelength,
            plot_apertures: bool = True,
            plot_vignetted: bool = False,
    ):
        if surf is None:
            surf = self.image_surface

        fig, ax = plt.subplots()

        rays = self.raytrace_subsystem(self.input_rays, final_surface=surf)

        rays.plot_position(ax=ax, color_axis=color_axis, plot_vignetted=plot_vignetted)

        if plot_apertures:
            surf.plot_2d(ax)

    def plot_projections(
            self,
            start_surface: typ.Optional[surface.Surface] = None,
            end_surface: typ.Optional[surface.Surface] = None,
            color_axis: int = 0,
            plot_vignetted: bool = False,
    ):
        fig, axs = plt.subplots(2, 2, sharex='col', sharey='row')

        xy = 0, 0
        yz = 0, 1
        xz = 1, 1

        axs[xy].invert_xaxis()

        ax_indices = [xy, yz, xz]
        planes = [
            (kgpy.vector.ix, kgpy.vector.iy),
            (kgpy.vector.iz, kgpy.vector.iy),
            (kgpy.vector.iz, kgpy.vector.ix),
        ]
        for ax_index, plane in zip(ax_indices, planes):
            self.plot_2d(
                ax=axs[ax_index],
                components=plane,
                start_surface=start_surface,
                end_surface=end_surface,
                color_axis=color_axis,
                plot_vignetted=plot_vignetted
            )

        handles, labels = axs[xy].get_legend_handles_labels()
        label_dict = dict(zip(labels, handles))
        fig.legend(label_dict.values(), label_dict.keys())

    def plot_2d(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[int, int] = (kgpy.vector.ix, kgpy.vector.iy),
            start_surface: typ.Optional[surface.Surface] = None,
            end_surface: typ.Optional[surface.Surface] = None,
            color_axis: int = Rays.axis.wavelength,
            plot_vignetted: bool = False,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()

        surfaces = list(self)  # type: typ.List[surface.Surface]

        if start_surface is None:
            start_surface = surfaces[0]

        if end_surface is None:
            end_surface = surfaces[~0]

        start_surface_index = surfaces.index(start_surface)
        end_surface_index = surfaces.index(end_surface)

        intercepts = []
        i = slice(start_surface_index, end_surface_index)
        all_rays = self.all_rays
        for surf, rays in zip(surfaces[i], all_rays[i]):
            surf.plot_2d(ax, components, self)
            intercept = surf.transform_to_global(rays.position, self, num_extra_dims=5)
            intercepts.append(intercept)
        intercepts = u.Quantity(intercepts)

        img_rays = all_rays[~0]

        color_axis = (color_axis % img_rays.axis.ndim) - img_rays.axis.ndim

        if plot_vignetted:
            mask = img_rays.error_mask & img_rays.field_mask
        else:
            mask = img_rays.mask

        grid = img_rays.input_grids[color_axis].flatten()
        colors = plt.cm.viridis((grid - grid.min()) / (grid.max() - grid.min()))
        labels = img_rays.grid_labels(color_axis).flatten()

        intercepts = np.moveaxis(intercepts, color_axis - 1, img_rays.ndim + 1)
        mask = np.moveaxis(mask, color_axis, img_rays.ndim)

        new_shape = intercepts.shape[0:1] + (-1,) + grid.shape + intercepts.shape[~(img_rays.vaxis.ndim - 2):]
        intercepts = intercepts.reshape(new_shape)
        mask = mask.reshape((-1,) + grid.shape + mask.shape[~(img_rays.axis.ndim - 2):])

        intercepts = np.moveaxis(intercepts, ~(img_rays.vaxis.ndim - 1), 0)
        mask = np.moveaxis(mask, ~(img_rays.axis.ndim - 1), 0)

        for intercept_c, mask_c, color, label in zip(intercepts, mask, colors, labels):
            ax.plot(
                intercept_c[:, mask_c, components[0]],
                intercept_c[:, mask_c, components[1]],
                color=color,
                label=label,
            )

        return ax

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

                        psh = list(surf.aperture.edges.shape)
                        psh[~0] = 3
                        polys = np.zeros(psh) * u.mm
                        polys[..., 0:2] = surf.aperture.edges

                        polys = self.local_to_global(surf, polys, configuration_axis=None)[0]

                        for poly in polys:
                            x_, y_, z_ = poly.T
                            ax.plot(x_, y_, z_, 'k')
                            ax.plot(u.Quantity([x_[-1], x_[0]]), u.Quantity([y_[-1], y_[0]]),
                                    u.Quantity([z_[-1], z_[0]]), 'k')

    def __iter__(self) -> typ.Iterator[surface.Surface]:
        yield from self.object_surface
        yield from self.surfaces
