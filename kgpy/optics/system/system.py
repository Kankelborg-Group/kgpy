import dataclasses
import pathlib
import pickle
import numpy as np
import typing as typ
import scipy.spatial.transform
import astropy.units as u
import astropy.visualization
import matplotlib.pyplot as plt
import kgpy.mixin
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

        # return np.broadcast(
        #     all_surface_battrs,
        #     # self.fields.config_broadcast,
        #     # self.wavelengths.config_broadcast,
        #     # self.entrance_pupil_radius,
        #     # self.stop_surface_index,
        # )

    def raytrace_subsystem(
            self,
            rays: Rays,
            start_surface: typ.Optional[surface.Surface] = None,
            final_surface: typ.Optional[surface.Surface] = None,
    ) -> Rays:

        surfaces = list(self)

        if start_surface is None:
            start_surface = surfaces[0]

        if final_surface is None:
            final_surface = surfaces[~0]

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

    def local_to_global(
            self,
            local_surface: surface.Surface,
            x: u.Quantity,
            extra_dim: bool = False
    ) -> u.Quantity:

        surfaces = list(self)
        for s, surf in enumerate(surfaces):
            if surf is local_surface:
                local_surface_index = s
                break
        surfaces = surfaces[:local_surface_index]
        surfaces.reverse()

        if isinstance(local_surface, surface.Standard):
            x = local_surface.transform_before.apply(x, extra_dim=extra_dim)

        for surf in surfaces:

            if isinstance(surf, surface.CoordinateBreak):
                if surf is not local_surface:
                    x = surf.transform.apply(x, extra_dim=extra_dim)

            elif isinstance(surf, surface.Standard):
                x = surf.transform_before.apply(x, extra_dim=extra_dim)
                x = surf.transform_after.apply(x, extra_dim=extra_dim)

            x[..., ~0] += surf.thickness

        return x

    def plot_footprint(self, surf: surface.Standard):

        pass

    def plot_projections(
            self,
            start_surface: typ.Optional[surface.Surface] = None,
            end_surface: typ.Optional[surface.Surface] = None,
    ):

        surfaces = list(self)

        start_surface_index = 0
        end_surface_index = None
        for s, surf in enumerate(surfaces):
            if surf is start_surface:
                start_surface_index = s
            if surf is end_surface:
                end_surface_index = s + 1
                break

        x = ..., 0
        y = ..., 1
        z = ..., 2

        with astropy.visualization.quantity_support():

            fig, axs = plt.subplots(2, 2, sharex='col', sharey='row')
            axs[0, 0].invert_xaxis()

            self.plot_surface_projections(axs)

            points = []
            i = slice(start_surface_index, end_surface_index)
            for s, r in zip(surfaces[i], self.all_rays[i]):
                if isinstance(s, surface.Standard):
                    points.append(self.local_to_global(s, r.position, extra_dim=True))
            points = u.Quantity(points)

            # points = u.Quantity([self.local_to_global(s, r.position, extra_dim=True) for s, r in zip(self, self.all_rays)])
            points = points.reshape((points.shape[0], -1, points.shape[~0]))

            axs[0, 0].plot(points[x], points[y])
            axs[0, 1].plot(points[z], points[y])
            axs[1, 1].plot(points[z], points[x])

    def plot_surface_projections(self, axs: np.ndarray):
        x = ..., 0
        y = ..., 1
        z = ..., 2

        prop_direction = 1
        for surf in self.surfaces:
            if isinstance(surf, surface.Standard):
                if not isinstance(surf.aperture, aperture.NoAperture):

                    points = surf.aperture.edges.copy()
                    points[z] = surf.sag(points[x], points[y])
                    points = self.local_to_global(surf, points, extra_dim=True)
                    points = points.to(u.mm)

                    axs[0, 0].fill(points[x].T, points[y].T, fill=False)
                    axs[0, 1].fill(points[z].T, points[y].T, fill=False)
                    axs[1, 1].fill(points[z].T, points[x].T, fill=False)

                    if isinstance(surf.material, material.Mirror):
                        back = surf.aperture.edges.copy()

                        xmax, ymax = back[x].max(~0, keepdims=True), back[y].max(~0, keepdims=True)
                        xmin, ymin = back[x].min(~0, keepdims=True), back[y].min(~0, keepdims=True)

                        t = prop_direction * surf.material.thickness
                        back[z] = t
                        prop_direction *= -1
                        back = self.local_to_global(surf, back, extra_dim=True)

                        t = np.broadcast_to(t, xmax.shape, subok=True)
                        c1 = np.concatenate([
                            np.stack([xmax, ymax, surf.sag(xmax, ymax)], axis=~0),
                            np.stack([xmax, ymax, t], axis=~0),
                        ], axis=~1)
                        c2 = np.concatenate([
                            np.stack([xmax, ymin, surf.sag(xmax, ymin)], axis=~0),
                            np.stack([xmax, ymin, t], axis=~0),
                        ], axis=~1)
                        c3 = np.concatenate([
                            np.stack([xmin, ymin, surf.sag(xmin, ymin)], axis=~0),
                            np.stack([xmin, ymin, t], axis=~0),
                        ], axis=~1)
                        c4 = np.concatenate([
                            np.stack([xmin, ymax, surf.sag(xmin, ymax)], axis=~0),
                            np.stack([xmin, ymax, t], axis=~0),
                        ], axis=~1)

                        c1 = self.local_to_global(surf, c1, extra_dim=True)
                        c2 = self.local_to_global(surf, c2, extra_dim=True)
                        c3 = self.local_to_global(surf, c3, extra_dim=True)
                        c4 = self.local_to_global(surf, c4, extra_dim=True)

                        axs[0, 0].fill(back[x].T, back[y].T, fill=False)
                        axs[0, 1].fill(back[z].T, back[y].T, fill=False)
                        axs[1, 1].fill(back[z].T, back[x].T, fill=False)

                        axs[0, 0].plot(c1[x].T, c1[y].T, color='black')
                        axs[0, 0].plot(c2[x].T, c2[y].T, color='black')
                        axs[0, 0].plot(c3[x].T, c3[y].T, color='black')
                        axs[0, 0].plot(c4[x].T, c4[y].T, color='black')

                        axs[0, 1].plot(c1[z].T, c1[y].T, color='black')
                        axs[0, 1].plot(c2[z].T, c2[y].T, color='black')
                        axs[0, 1].plot(c3[z].T, c3[y].T, color='black')
                        axs[0, 1].plot(c4[z].T, c4[y].T, color='black')

                        axs[1, 1].plot(c1[z].T, c1[x].T, color='black')
                        axs[1, 1].plot(c2[z].T, c2[x].T, color='black')
                        axs[1, 1].plot(c3[z].T, c3[x].T, color='black')
                        axs[1, 1].plot(c4[z].T, c4[x].T, color='black')

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
