"""
kgpy.optics is a package designed for simulating optical systems.
"""
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
from kgpy import mixin, linspace, vector, optimization, transform
from kgpy.vector import x, y, z, ix, iy, iz, xy
from . import aberration, rays, surface, component

__all__ = [
    'aberration',
    'rays',
    'surface',
    'component',
    'System'
]


@dataclasses.dataclass
class System(
    transform.rigid.Transformable,
    mixin.Broadcastable,
    mixin.Named,
):
    """
    Model of an optical system.
    """
    #: Surface representing the light source
    object_surface: surface.Surface = dataclasses.field(default_factory=surface.Surface)
    surfaces: surface.SurfaceList = dataclasses.field(default_factory=surface.SurfaceList)
    wavelengths: u.Quantity = 0 * u.nm    #: Source wavelengths
    pupil_samples: typ.Union[int, typ.Tuple[int, int]] = 3      #: Number of samples across the pupil for each axis x, y
    pupil_margin: u.Quantity = 1 * u.um     #: Margin between edge of pupil and nearest ray
    field_samples: typ.Union[int, typ.Tuple[int, int]] = 3      #: Number of samples across the field for each axis x, y
    field_margin: u.Quantity = 1 * u.nrad       #: Margin between edge of field and nearest ray
    # baffle_positions: typ.Optional[typ.List[transform.rigid.Transform]] = None

    def __post_init__(self):
        self.update()

    def update(self) -> typ.NoReturn:
        self._rays_input_cache = None
        self._raytrace_cache = None

    @property
    def stop(self) -> surface.Surface:
        return [s for s in self.surfaces if s.is_stop][0]

    @property
    def stop_tests(self) -> typ.List[surface.Surface]:
        return [s for s in self.surfaces if s.is_stop_test]

    @property
    def surfaces_all(self) -> surface.SurfaceList:
        return surface.SurfaceList([self.object_surface]) + self.surfaces

    @property
    def aperture_surfaces(self) -> typ.Iterator[surface.Surface]:
        for s in self.surfaces:
            if s.aperture is not None:
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
    def field_min(self) -> u.Quantity:
        return self.object_surface.aperture.min

    @property
    def field_max(self) -> u.Quantity:
        return self.object_surface.aperture.max

    @property
    def field_x(self) -> u.Quantity:
        return linspace(
            start=self.field_min[x] + self.field_margin,
            stop=self.field_max[x] - self.field_margin,
            num=self.field_samples_normalized[ix],
            axis=~0,
        )

    @property
    def field_y(self) -> u.Quantity:
        return linspace(
            start=self.field_min[y] + self.field_margin,
            stop=self.field_max[y] - self.field_margin,
            num=self.field_samples_normalized[iy],
            axis=~0
        )

    def pupil_x(self, surf: surface.Surface) -> u.Quantity:
        aper = surf.aperture
        return linspace(
            start=aper.min[x] + self.pupil_margin,
            stop=aper.max[x] - self.pupil_margin,
            num=self.pupil_samples_normalized[ix],
            axis=~0,
        )

    def pupil_y(self, surf: surface.Surface) -> u.Quantity:
        aper = surf.aperture
        return linspace(
            start=aper.min[y] + self.pupil_margin,
            stop=aper.max[y] - self.pupil_margin,
            num=self.pupil_samples_normalized[iy],
            axis=~0,
        )

    @property
    def raytrace(self) -> rays.RaysList:
        if self._raytrace_cache is None:
            self._update_raytrace_caches()
        return self._raytrace_cache

    @property
    def _raytrace(self) -> rays.RaysList:
        return self.surfaces_all.raytrace(self.rays_input)

    @property
    def rays_output(self) -> rays.Rays:
        return self.raytrace[~0]

    @property
    def rays_input(self):
        return self.raytrace[0]

    def _update_raytrace_caches(self) -> typ.NoReturn:

        stops = self.stop_tests + [self.stop]

        for surf in stops:

            if self.field_min.unit.is_equivalent(u.rad):

                if self._rays_input_cache is None:
                    position_guess = vector.from_components(use_z=False) << u.mm
                else:
                    position_guess = self._rays_input_cache.position[xy]

                step_size = .1 * u.mm
                step = vector.from_components(x=step_size, y=step_size, use_z=False)

                # surf = self.stop
                stop_index = self.surfaces_all.index(surf)
                px, py = self.pupil_x(surf), self.pupil_y(surf)
                target_position = vector.from_components(px[..., None], py)

                def position_error(pos: u.Quantity) -> u.Quantity:
                    position = vector.to_3d(pos)
                    rays_in = rays.Rays.from_field_angles(
                        wavelength_grid=self.wavelengths,
                        position=position,
                        field_grid_x=self.field_x,
                        field_grid_y=self.field_y,
                        pupil_grid_x=px,
                        pupil_grid_y=py
                    )
                    rays_in.transform = self.object_surface.transform
                    raytrace = self.surfaces_all.raytrace(rays_in)

                    self._rays_input_cache = rays_in
                    self._raytrace_cache = raytrace

                    return (raytrace[stop_index].position - target_position)[xy]

                optimization.root_finding.vector.secant(
                    func=position_error,
                    root_guess=position_guess,
                    step_size=step,
                    max_abs_error=1 * u.nm,
                    max_iterations=100,
                )

            else:

                if self._rays_input_cache is None:
                    direction_guess = vector.from_components(use_z=False) << u.deg
                else:
                    direction_guess = self._rays_input_cache.direction

                step_size = 1e-10 * u.deg
                step = vector.from_components(x=step_size, y=step_size, use_z=False)

                # surf = self.stop
                stop_index = self.surfaces_all.index(surf)
                px, py = self.pupil_x(surf), self.pupil_y(surf)
                target_position = vector.from_components(px[..., None], py)

                def position_error(direc: u.Quantity) -> u.Quantity:
                    direction = np.zeros(self.field_samples_normalized + target_position.shape)
                    direction[z] = 1
                    direction = transform.rigid.TiltX(direc[y])(direction)
                    direction = transform.rigid.TiltY(direc[x])(direction)
                    rays_in = rays.Rays.from_field_positions(
                        wavelength_grid=self.wavelengths,
                        direction=direction,
                        field_grid_x=self.field_x,
                        field_grid_y=self.field_y,
                        pupil_grid_x=px,
                        pupil_grid_y=py,
                    )
                    rays_in.transform = self.object_surface.transform
                    raytrace = self.surfaces_all.raytrace(rays_in)

                    self._rays_input_cache = rays_in
                    self._raytrace_cache = raytrace

                    return (raytrace[stop_index].position - target_position)[xy]

                optimization.root_finding.vector.secant(
                    func=position_error,
                    root_guess=direction_guess,
                    step_size=step,
                    max_abs_error=1 * u.nm,
                    max_iterations=100,
                )

    def psf(
            self,
            bins: typ.Union[int, typ.Tuple[int, int]] = 10,
            limits: typ.Optional[typ.Tuple[typ.Tuple[int, int], typ.Tuple[int, int]]] = None,
            use_vignetted: bool = False,
            relative_to_centroid: typ.Tuple[bool, bool] = (False, False),
    ) -> typ.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.rays_output.pupil_hist2d(
            bins=bins,
            limits=limits,
            use_vignetted=use_vignetted,
            relative_to_centroid=relative_to_centroid,
        )

    def print_surfaces(self) -> typ.NoReturn:
        for surf in self.surfaces_all:
            print(surf)

    @property
    def broadcasted(self):
        all_surface_battrs = None
        for s in self.surfaces:
            all_surface_battrs = np.broadcast(all_surface_battrs, s.broadcasted)
            all_surface_battrs = np.broadcast_to(np.array(1), all_surface_battrs.shape)

        return all_surface_battrs

    def plot_footprint(
            self,
            ax: typ.Optional[plt.Axes] = None,
            surf: typ.Optional[surface.Surface] = None,
            color_axis: int = rays.Rays.axis.wavelength,
            plot_apertures: bool = True,
            plot_vignetted: bool = False,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()

        if surf is None:
            surf = self.surfaces[~0]

        surf_index = self.surfaces_all.index(surf)
        rays = self.raytrace[surf_index].copy()
        rays.vignetted_mask = self.rays_output.vignetted_mask

        rays.plot_position(ax=ax, color_axis=color_axis, plot_vignetted=plot_vignetted)

        if plot_apertures:
            surf.plot(ax=ax)

        return ax

    def plot_projections(
            self,
            surface_first: typ.Optional[surface.Surface] = None,
            surface_last: typ.Optional[surface.Surface] = None,
            color_axis: int = 0,
            plot_vignetted: bool = False,
            plot_rays: bool = True,
    ) -> plt.Figure:
        fig, axs = plt.subplots(2, 2, sharex='col', sharey='row')

        xy = 0, 0
        yz = 0, 1
        xz = 1, 1

        axs[xy].invert_xaxis()

        ax_indices = [xy, yz, xz]
        planes = [
            (vector.ix, vector.iy),
            (vector.iz, vector.iy),
            (vector.iz, vector.ix),
        ]
        for ax_index, plane in zip(ax_indices, planes):
            self.plot(
                ax=axs[ax_index],
                components=plane,
                surface_first=surface_first,
                surface_last=surface_last,
                color_axis=color_axis,
                plot_vignetted=plot_vignetted,
                plot_rays=plot_rays,
            )
            if plot_rays:
                axs[ax_index].get_legend().remove()

        handles, labels = axs[xy].get_legend_handles_labels()
        label_dict = dict(zip(labels, handles))
        fig.legend(label_dict.values(), label_dict.keys())

        return fig

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[int, int] = (vector.ix, vector.iy),
            rigid_transform: typ.Optional[transform.rigid.TransformList] = None,
            surface_first: typ.Optional[surface.Surface] = None,
            surface_last: typ.Optional[surface.Surface] = None,
            plot_rays: bool = True,
            color_axis: int = rays.Rays.axis.wavelength,
            plot_vignetted: bool = False,
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()

        if surface_first is None:
            surface_first = self.surfaces_all[0]
        if surface_last is None:
            surface_last = self.surfaces_all[~0]
        surface_index_first = self.surfaces_all.index(surface_first)
        surface_index_last = self.surfaces_all.index(surface_last)

        surf_slice = slice(surface_index_first, surface_index_last + 1)

        if plot_rays:
            raytrace_slice = self.raytrace[surf_slice]      # type: RaysList
            raytrace_slice.plot(
                ax=ax,
                components=components,
                rigid_transform=rigid_transform,
                color_axis=color_axis,
                plot_vignetted=plot_vignetted,
            )

        surfaces_slice = self.surfaces_all[surf_slice]  # type: SurfaceList
        surfaces_slice.plot(
            ax=ax,
            components=components,
            rigid_transform=rigid_transform,
        )

        return ax










