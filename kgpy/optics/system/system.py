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
from .. import ZemaxCompatible, Rays, material, surface, aperture

__all__ = ['System']

SurfacesT = typ.TypeVar('SurfacesT', bound=typ.Union[typ.Iterable[surface.Surface], ZemaxCompatible])


@dataclasses.dataclass
class System(ZemaxCompatible, kgpy.mixin.Broadcastable, kgpy.mixin.Named, typ.Generic[SurfacesT]):
    object_surface: surface.ObjectSurface = dataclasses.field(default_factory=lambda: surface.ObjectSurface())
    surfaces: SurfacesT = dataclasses.field(default_factory=lambda: [surface.Standard()])
    stop_surface: surface.Surface = None
    input_rays: Rays = dataclasses.field(default_factory=lambda: Rays.zeros())

    def __post_init__(self):
        if self.stop_surface is None:
            self.stop_surface = next(iter(self.surfaces))

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
    def image_surface(self) -> surface.Surface:
        s = list(self)
        return s[~0]

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
    ) -> typ.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.image_rays.pupil_hist2d(
            bins=bins,
            limits=limits,
            use_vignetted=use_vignetted,
            relative_to_centroid=True,
        )

    def plot_footprint(
            self,
            surf: typ.Optional[surface.Standard] = None,
            config_index: typ.Union[int, slice] = slice(None),
            color_axis: int = 0,
            plot_apertures: bool = True,
            plot_vignetted: bool = False,
    ):
        if surf is None:
            surf = self.image_surface

        rays = self.raytrace_subsystem(self.input_rays, final_surface=surf)
        position = rays.position[config_index]

        if self.shape:
            sh = [1] * rays.ndim
            sh[color_axis] = rays.shape[color_axis]
            colors = np.arange(rays.shape[color_axis]).reshape(sh)
            colors = np.broadcast_to(colors, rays.shape)
            colors = colors[config_index]
        else:
            colors = None

        if not plot_vignetted:
            mask = rays.unvignetted_mask[config_index]
            px = position[mask, kgpy.vector.ix]
            py = position[mask, kgpy.vector.iy]
            colors = colors[mask]
        else:
            px = position[kgpy.vector.x]
            py = position[kgpy.vector.y]

        with astropy.visualization.quantity_support():

            fig, ax = plt.subplots()

            if plot_apertures:
                surf.plot_2d(ax)

            ax.scatter(
                x=px,
                y=py,
                c=colors,
            )

    def plot_psf_vs_field(
            self,
            config_index: int = 0,
            wavlen_index: int = 0,
            bins: typ.Union[int, typ.Tuple[int, int]] = 10,
            limits: typ.Optional[typ.Tuple[typ.Tuple[int, int], typ.Tuple[int, int]]] = None,
            use_vignetted: bool = False,
    ) -> plt.Figure:

        rays = self.input_rays

        fax_x = (rays.axis.field_y, rays.axis.pupil_x, rays.axis.pupil_y)
        field_x = rays.direction[config_index, wavlen_index, ..., kgpy.vector.ix].mean(fax_x)
        field_x = np.arcsin(field_x) << u.rad

        fax_y = (rays.axis.field_x, rays.axis.pupil_x, rays.axis.pupil_y)
        field_y = rays.direction[config_index, wavlen_index, ..., kgpy.vector.iy].mean(fax_y)
        field_y = np.arcsin(field_y) << u.rad

        return self.image_rays.plot_pupil_hist2d_vs_field(
            config_index=config_index,
            wavlen_index=wavlen_index,
            bins=bins,
            limits=limits,
            use_vignetted=use_vignetted,
            field_x=field_x,
            field_y=field_y,
        )

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

    def plot_2d(
            self,
            ax: plt.Axes,
            components: typ.Tuple[int, int] = (kgpy.vector.ix, kgpy.vector.iy),
            start_surface: typ.Optional[surface.Surface] = None,
            end_surface: typ.Optional[surface.Surface] = None,
            color_axis: int = 0,
            plot_vignetted: bool = False,
    ):
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

        mask = ~all_rays[~0].error_mask
        if not plot_vignetted:
            mask &= all_rays[~0].unvignetted_mask

        if self.shape:
            sh = [1] * intercepts.ndim
            sh[color_axis] = intercepts.shape[color_axis]
            colors = np.arange(intercepts.shape[color_axis]).reshape(sh)
            colors = np.broadcast_to(colors, intercepts.shape)
            colors = colors[:, mask, 0]
        else:
            colors = None

        intercepts = intercepts[:, mask, :]
        ax.plot(
            intercepts[..., components[0]],
            intercepts[..., components[1]],
            # color=colors,
        )

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
