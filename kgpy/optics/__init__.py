"""
kgpy.optics is a package designed for simulating optical systems.
"""
import collections
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
from . import aberration, rays, surface, component, baffle, breadboard

__all__ = [
    'aberration',
    'rays',
    'surface',
    'component',
    'baffle',
    'breadboard',
    'System',
    'SystemList',
]


@dataclasses.dataclass
class System(
    transform.rigid.Transformable,
    mixin.Broadcastable,
    mixin.Named,
):
    from .breadboard import Breadboard
    """
    Model of an optical system.
    """
    #: Surface representing the light source
    object_surface: surface.Surface = dataclasses.field(default_factory=surface.Surface)
    surfaces: surface.SurfaceList = dataclasses.field(default_factory=surface.SurfaceList)
    wavelengths: u.Quantity = 0 * u.nm  #: Source wavelengths
    pupil_samples: typ.Union[int, typ.Tuple[int, int]] = 3  #: Number of samples across the pupil for each axis x, y
    pupil_margin: u.Quantity = 1 * u.nm  #: Margin between edge of pupil and nearest ray
    field_samples: typ.Union[int, typ.Tuple[int, int]] = 3  #: Number of samples across the field for each axis x, y
    field_margin: u.Quantity = 1 * u.nrad  #: Margin between edge of field and nearest ray
    pointing: vector.Vector2D = dataclasses.field(default_factory=lambda: vector.Vector2D(x=0 * u.deg, y=0 * u.deg))
    roll: u.Quantity = 0 * u.deg
    baffles_blank: baffle.BaffleList = dataclasses.field(default_factory=baffle.BaffleList)
    baffles_hull_axes: typ.Optional[typ.Tuple[int, ...]] = None
    breadboard: typ.Optional[Breadboard] = None
    tolerance_axes: typ.Dict[str, int] = dataclasses.field(default_factory=lambda: {})
    focus_axes: typ.Dict[str, int] = dataclasses.field(default_factory=lambda: {})

    def __post_init__(self):
        self.update()

    def update(self) -> typ.NoReturn:
        self._rays_input_cache = None
        self._raytrace_cache = None
        self._baffles_cache = None

    @property
    def surfaces_all(self) -> surface.SurfaceList:
        return surface.SurfaceList([self.object_surface]) + self.surfaces

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
    def field_min(self) -> vector.Vector2D:
        return self.object_surface.aperture.min.xy

    @property
    def field_max(self) -> vector.Vector2D:
        return self.object_surface.aperture.max.xy

    @property
    def field_x(self) -> u.Quantity:
        return linspace(
            start=self.field_min.x + self.field_margin,
            stop=self.field_max.x - self.field_margin,
            num=self.field_samples_normalized[vector.ix],
            axis=~0,
        )

    @property
    def field_y(self) -> u.Quantity:
        return linspace(
            start=self.field_min.y + self.field_margin,
            stop=self.field_max.y - self.field_margin,
            num=self.field_samples_normalized[vector.iy],
            axis=~0
        )

    def pupil_x(self, surf: surface.Surface) -> u.Quantity:
        aper = surf.aperture
        return linspace(
            start=aper.min.x + self.pupil_margin,
            stop=aper.max.x - self.pupil_margin,
            num=self.pupil_samples_normalized[vector.ix],
            axis=~0,
        )

    def pupil_y(self, surf: surface.Surface) -> u.Quantity:
        aper = surf.aperture
        return linspace(
            start=aper.min.y + self.pupil_margin,
            stop=aper.max.y - self.pupil_margin,
            num=self.pupil_samples_normalized[vector.iy],
            axis=~0,
        )

    @property
    def baffle_lofts(self) -> typ.Dict[int, typ.Tuple[surface.Surface, surface.Surface]]:
        lofts = {}
        for surf in self.surfaces_all.flat_global_iter:
            for b_id in surf.baffle_loft_ids:
                if b_id not in lofts:
                    lofts[b_id] = ()
                if len(lofts[b_id]) >= 2:
                    raise ValueError('Loft with more than two surfaces')
                lofts[b_id] = lofts[b_id] + (surf, )

        return lofts

    @property
    def baffles(self) -> baffle.BaffleList:
        if self._baffles_cache is None:
            self._baffles_cache = self.calc_baffles(self.baffles_blank)
        return self._baffles_cache

    def calc_baffles(
            self,
            baffles_blank: baffle.BaffleList,
            transform_extra: typ.Optional[transform.rigid.TransformList] = None,
    ) -> baffle.BaffleList:

        if transform_extra is None:
            transform_extra = transform.rigid.TransformList()

        if baffles_blank:

            baffles = baffles_blank
            baffles = baffles.concat_apertures_from_raytrace(
                raytrace=self.raytrace, transform_extra=transform_extra, hull_axes=self.baffles_hull_axes, color='red')

            baffles = baffles.concat_apertures_from_lofts(lofts=self.baffle_lofts, transform_extra=transform_extra)

        else:
            baffles = baffles_blank

        return baffles

    @property
    def raytrace(self) -> rays.RaysList:
        if self._raytrace_cache is None:
            self._raytrace_cache = self.surfaces_all.raytrace(self.rays_input)
        return self._raytrace_cache

    @property
    def rays_output(self) -> rays.Rays:
        return self.raytrace[~0]

    @property
    def rays_input(self) -> rays.Rays:
        if self._rays_input_cache is None:
            self._rays_input_cache = self._calc_rays_input()
        return self._rays_input_cache

    def _calc_rays_input(self) -> rays.Rays:

        rays_input = None

        for surf_index, surf in enumerate(self.surfaces_all.flat_global_iter):

            if not surf.is_stop and not surf.is_stop_test:
                continue

            if self.field_min.quantity.unit.is_equivalent(u.rad):

                if rays_input is None:
                    position_guess = vector.Vector2D.spatial()
                else:
                    position_guess = rays_input.position.xy

                step_size = .1 * u.mm

                px, py = self.pupil_x(surf), self.pupil_y(surf)
                # px = np.expand_dims(px, rays.Rays.axis.perp_axes(rays.Rays.axis.pupil_x))
                # py = np.expand_dims(py, rays.Rays.axis.perp_axes(rays.Rays.axis.pupil_y))
                target_position = vector.Vector2D(
                    x=np.expand_dims(px, rays.Rays.axis.perp_axes(rays.Rays.axis.pupil_x)),
                    y=np.expand_dims(py, rays.Rays.axis.perp_axes(rays.Rays.axis.pupil_y)),
                )

                def position_error(pos: vector.Vector2D) -> vector.Vector2D:
                    rays_in = rays.Rays.from_field_angles(
                        wavelength_grid=self.wavelengths,
                        position=pos.to_3d(),
                        field_grid_x=self.field_x,
                        field_grid_y=self.field_y,
                        pupil_grid_x=px,
                        pupil_grid_y=py
                    )
                    rays_in.transform = self.object_surface.transform
                    raytrace = self.surfaces_all.raytrace(rays_in, surface_last=surf)

                    return raytrace[~0].position.xy - target_position

                position_final = optimization.root_finding.vector.secant_2d(
                    func=position_error,
                    root_guess=position_guess,
                    step_size=step_size,
                    max_abs_error=1 * u.nm,
                    max_iterations=100,
                )
                rays_input = rays.Rays.from_field_angles(
                    wavelength_grid=self.wavelengths,
                    position=position_final.to_3d(),
                    field_grid_x=self.field_x,
                    field_grid_y=self.field_y,
                    pupil_grid_x=px,
                    pupil_grid_y=py
                )
                rays_input.transform = self.object_surface.transform

            else:
                if rays_input is None:
                    direction_guess = vector.Vector2D(x=0 * u.deg, y=0 * u.deg)
                else:
                    direction_guess = np.arcsin(rays_input.direction.xy)

                step_size = 1e-10 * u.deg

                px, py = self.pupil_x(surf), self.pupil_y(surf)
                # px = np.expand_dims(px, rays.Rays.axis.perp_axes(rays.Rays.axis.pupil_x))
                # py = np.expand_dims(py, rays.Rays.axis.perp_axes(rays.Rays.axis.pupil_y))
                target_position = vector.Vector2D(
                    x=np.expand_dims(px, rays.Rays.axis.perp_axes(rays.Rays.axis.pupil_x)),
                    y=np.expand_dims(py, rays.Rays.axis.perp_axes(rays.Rays.axis.pupil_y)),
                )

                def position_error(angles: vector.Vector2D) -> vector.Vector2D:
                    direction = transform.rigid.TiltX(angles.y)(vector.z_hat)
                    direction = transform.rigid.TiltY(angles.x)(direction)
                    rays_in = rays.Rays.from_field_positions(
                        wavelength_grid=self.wavelengths,
                        direction=direction,
                        field_grid_x=self.field_x,
                        field_grid_y=self.field_y,
                        pupil_grid_x=px,
                        pupil_grid_y=py,
                    )
                    rays_in.transform = self.object_surface.transform
                    raytrace = self.surfaces_all.raytrace(rays_in, surface_last=surf)

                    return raytrace[~0].position.xy - target_position

                angles_final = optimization.root_finding.vector.secant_2d(
                    func=position_error,
                    root_guess=direction_guess,
                    step_size=step_size,
                    max_abs_error=1 * u.nm,
                    max_iterations=100,
                )
                direction_final = transform.rigid.TiltX(angles_final.y)(vector.z_hat)
                direction_final = transform.rigid.TiltY(angles_final.x)(direction_final)
                rays_input = rays.Rays.from_field_positions(
                    wavelength_grid=self.wavelengths,
                    direction=direction_final,
                    field_grid_x=self.field_x,
                    field_grid_y=self.field_y,
                    pupil_grid_x=px,
                    pupil_grid_y=py,
                )
                rays_input.transform = self.object_surface.transform

            if surf.is_stop:
                return rays_input

        raise ValueError('No stop defined')

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

    def __eq__(self, other: 'System'):
        if self.object_surface != other.object_surface:
            return True
        if self.surfaces != other.surfaces:
            return False
        if np.array(self.wavelengths != other.wavelengths).any():
            return False
        if self.pupil_samples != other.pupil_samples:
            return False
        if self.pupil_margin != other.pupil_margin:
            return False
        if self.field_samples != other.field_samples:
            return False
        if self.field_margin != other.field_margin:
            return False
        if self.baffles_blank != other.baffles_blank:
            return False
        if self.baffles_hull_axes != other.baffles_hull_axes:
            return False
        return True

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

        surfaces = self.surfaces_all.flat_local

        if surf is None:
            surf = surfaces[~0]

        surf_index = surfaces.index(surf)
        surf_rays = self.raytrace[surf_index].view()
        surf_rays.vignetted_mask = self.rays_output.vignetted_mask

        surf_rays.plot_position(ax=ax, color_axis=color_axis, plot_vignetted=plot_vignetted)

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
            ('x', 'y'),
            ('z', 'y'),
            ('z', 'x'),
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
        fig.legend(label_dict.values(), label_dict.keys(), loc='top left', bbox_to_anchor=(1.0, 1.0))

        return fig

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[str, str] = ('x', 'y'),
            transform_extra: typ.Optional[transform.rigid.TransformList] = None,
            surface_first: typ.Optional[surface.Surface] = None,
            surface_last: typ.Optional[surface.Surface] = None,
            plot_rays: bool = True,
            color_axis: int = rays.Rays.axis.wavelength,
            plot_vignetted: bool = False,
            plot_baffles: bool = True,
            plot_breadboard: bool = True,
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()

        surfaces = self.surfaces_all.flat_local

        if transform_extra is None:
            transform_extra = transform.rigid.TransformList()
        transform_extra = transform_extra + self.transform

        if surface_first is None:
            surface_first = surfaces[0]
        if surface_last is None:
            surface_last = surfaces[~0]
        surface_index_first = surfaces.index(surface_first)
        surface_index_last = surfaces.index(surface_last)

        surf_slice = slice(surface_index_first, surface_index_last + 1)

        if plot_rays:
            raytrace_slice = self.raytrace[surf_slice]  # type: rays.RaysList
            raytrace_slice.plot(
                ax=ax,
                components=components,
                transform_extra=transform_extra,
                color_axis=color_axis,
                plot_vignetted=plot_vignetted,
            )

        surfaces_slice = self.surfaces_all.flat_global[surf_slice]  # type: surfaces.SurfaceList
        surfaces_slice.plot(
            ax=ax,
            components=components,
            transform_extra=transform_extra,
            to_global=True,
        )

        if plot_baffles:
            if self.baffles is not None:
                self.baffles.plot(ax=ax, components=components, transform_extra=transform_extra)

        if plot_breadboard:
            if self.breadboard is not None:
                self.breadboard.plot(
                    ax=ax,
                    components=components,
                    transform_extra=transform_extra,
                    to_global=True,
                )

        return ax

    @property
    def tol_iter(self) -> typ.Iterator['System']:
        others = super().tol_iter   # type: typ.Iterator
        for other in others:
            for surfaces in other.surfaces.tol_iter:
                new_other = other.view()
                new_other.surfaces = surfaces
                yield new_other

    def view(self) -> 'System':
        other = super().view()  # type: System
        other.object_surface = self.object_surface
        other.surfaces = self.surfaces
        other.wavelengths = self.wavelengths
        other.pupil_samples = self.pupil_samples
        other.pupil_margin = self.pupil_margin
        other.field_samples = self.field_samples
        other.field_margin = self.field_margin
        other.pointing = self.pointing
        other.roll = self.roll
        other.baffles_blank = self.baffles_blank
        other.baffles_hull_axes = self.baffles_hull_axes
        other.breadboard = self.breadboard
        return other

    def copy(self) -> 'System':
        other = super().copy()  # type: System
        other.object_surface = self.object_surface.copy()
        other.surfaces = self.surfaces.copy()
        other.wavelengths = self.wavelengths.copy()
        other.pupil_samples = self.pupil_samples
        other.pupil_margin = self.pupil_margin.copy()
        other.field_samples = self.field_samples
        other.field_margin = self.field_margin.copy()
        other.pointing = self.pointing.copy()
        other.roll = self.roll.copy()
        other.baffles_blank = self.baffles_blank.copy()
        other.baffles_hull_axes = self.baffles_hull_axes
        if self.breadboard is None:
            other.breadboard = self.breadboard
        else:
            other.breadboard = self.breadboard.copy()
        return other


@dataclasses.dataclass
class SystemList(
    mixin.DataclassList[System],
):
    baffles_blank: baffle.BaffleList = dataclasses.field(default_factory=baffle.BaffleList)

    def __post_init__(self):
        self.update()

    def update(self) -> typ.NoReturn:
        self._baffles_cache = None

    @property
    def baffles(self) -> baffle.BaffleList:
        if self._baffles_cache is None:
            baffles = self.baffles_blank.copy()
            if baffles:
                for sys in self:
                    new_baffles = sys.calc_baffles(
                        baffles_blank=self.baffles_blank,
                        transform_extra=sys.transform,
                    )
                    baffles = baffle.BaffleList([b1.unary_union(b2) for b1, b2 in zip(baffles, new_baffles)])
            self._baffles_cache = baffles
        return self._baffles_cache

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[str, str] = ('x', 'y'),
            transform_extra: typ.Optional[transform.rigid.TransformList] = None,
            plot_rays: bool = True,
            color_axis: int = rays.Rays.axis.wavelength,
            plot_vignetted: bool = False,
            plot_baffles: bool = True,
    ) -> plt.Axes:

        if ax is None:
            _, ax = plt.subplots()

        for sys in self:
            sys.plot(
                ax=ax,
                components=components,
                transform_extra=transform_extra,
                plot_rays=plot_rays,
                color_axis=color_axis,
                plot_vignetted=plot_vignetted,
                plot_baffles=plot_baffles
            )

        if plot_baffles:
            self.baffles.plot(ax=ax, components=components, transform_extra=transform_extra)

        return ax
