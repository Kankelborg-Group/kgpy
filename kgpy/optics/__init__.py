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
import scipy.signal
import scipy.optimize
import scipy.interpolate
import astropy.units as u
import astropy.visualization
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.lines
import matplotlib.colorbar
import matplotlib.axes
from kgpy import mixin, linspace, vector, optimization, transform, obs, grid
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
    mixin.Plottable,
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
    grid_rays: rays.RayGrid = dataclasses.field(default_factory=rays.RayGrid)
    field_margin: u.Quantity = 1 * u.nrad  #: Margin between edge of field and nearest ray
    pupil_margin: u.Quantity = 1 * u.nm  #: Margin between edge of pupil and nearest ray
    pointing: vector.Vector2D = dataclasses.field(default_factory=vector.Vector2D.angular)
    roll: u.Quantity = 0 * u.deg
    baffles_blank: baffle.BaffleList = dataclasses.field(default_factory=baffle.BaffleList)
    baffles_hull_axes: typ.Optional[typ.Tuple[int, ...]] = None
    breadboard: typ.Optional[Breadboard] = None
    tolerance_axes: typ.Dict[str, int] = dataclasses.field(default_factory=lambda: {})
    focus_axes: typ.Dict[str, int] = dataclasses.field(default_factory=lambda: {})
    distortion_polynomial_degree: int = 2
    vignetting_polynomial_degree: int = 1

    def __post_init__(self):
        self.update()

    def update(self) -> typ.NoReturn:
        self._rays_input_cache = None
        self._rays_input_resample_entrance_cache = None
        self._raytrace_cache = None
        self._raytrace_resample_entrance_cache = None
        self._baffles_cache = None

    @property
    def transform_pointing(self) -> transform.rigid.TransformList:
        return transform.rigid.TransformList([
            transform.rigid.TiltX(self.pointing.y),
            transform.rigid.TiltY(self.pointing.x),
        ])

    @property
    def transform_roll(self) -> transform.rigid.TransformList:
        return transform.rigid.TransformList([
            transform.rigid.TiltZ(self.roll)
        ])

    @property
    def transform_all(self) -> transform.rigid.TransformList:
        return self.transform_pointing + self.transform_roll + self.transform

    @property
    def surfaces_all(self) -> surface.SurfaceList:
        surfaces = self.surfaces.copy()
        surfaces.transform = self.transform_all
        return surface.SurfaceList([self.object_surface, surfaces])
        # surfaces = surface.SurfaceList([self.object_surface]) + self.surfaces
        # surfaces.transform = self.transform_all
        # return surfaces

    @property
    def surface_stop(self):
        for surf in self.surfaces_all.flat_global_iter:
            if surf.is_stop:
                return surf
        raise self.error_no_stop

    @property
    def wavelength(self) -> u.Quantity:
        return self.grid_wavelength.points

    def _calc_grid_rays(self, surf: surface.Surface) -> rays.RayGrid:
        grid_rays = self.grid_rays.copy()
        grid_rays.field.min = self.object_surface.aperture.min.xy + self.field_margin
        grid_rays.field.max = self.object_surface.aperture.max.xy - self.field_margin
        grid_rays.pupil.min = surf.aperture.min.xy + self.pupil_margin
        grid_rays.pupil.max = surf.aperture.max.xy - self.pupil_margin
        return grid_rays

    @property
    def grid_rays_stop(self) -> rays.RayGrid:
        return self._calc_grid_rays(self.surface_stop)

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
            self._rays_input_cache = self._calc_rays_input(self.grid_rays_stop)
        return self._rays_input_cache

    @property
    def rays_input_resample_entrance(self) -> rays.Rays:
        if self._rays_input_resample_entrance_cache is None:
            rays_input = self.rays_input.copy()
            rays_output = self.rays_output
            if rays_input.input_grid.field.points.x.unit.is_equivalent(u.rad):
                position = np.broadcast_to(rays_input.position, rays_output.position.shape, subok=True).copy()
                position[~rays_output.mask] = np.nan
                rays_input.input_grid.pupil.min = np.nanmin(position, axis=tuple(rays_output.axis.all))
                rays_input.input_grid.pupil.max = np.nanmax(position, axis=tuple(rays_output.axis.all))
                rays_input.position = rays_input.input_grid.points_pupil.to_3d()
                rays_input.intensity = rays_input.input_grid.pupil.step_size.x * rays_input.input_grid.pupil.step_size.y
                self._rays_input_resample_entrance_cache = rays_input
            else:
                raise NotImplementedError
        return self._rays_input_resample_entrance_cache

    @property
    def raytrace_resample_entrance(self) -> rays.RaysList:
        if self._raytrace_resample_entrance_cache is None:
            self._raytrace_resample_entrance_cache = self.surfaces_all.raytrace(self.rays_input_resample_entrance)
        return self._raytrace_resample_entrance_cache

    @property
    def rays_output_resample_entrance(self) -> rays.Rays:
        return self.raytrace_resample_entrance[~0]

    error_no_stop = ValueError('no stop defined')

    def _calc_rays_input_position(self, rays_input: rays.Rays) -> rays.Rays:
        rays_input = rays_input.copy()
        surfaces_all_global = self.surfaces_all.flat_global
        # rays_input.transform = self.transform + self.object_surface.transform
        rays_input.transform = surfaces_all_global[0].transform
        # for surf_index, surf in enumerate(self.surfaces_all.flat_global_iter):
        for surf_index, surf in enumerate(surfaces_all_global):
            if surf.is_stop or surf.is_stop_test:
                grid_surf = self._calc_grid_rays(surf)
                target_position = grid_surf.points_pupil

                def position_error(pos: vector.Vector2D) -> vector.Vector2D:
                    rays_in = rays_input.view()
                    rays_in.position = pos.to_3d()
                    rays_in.input_grid.pupil = grid_surf.pupil
                    raytrace = self.surfaces_all.raytrace(rays_in, surface_last=surf)
                    return raytrace[~0].position.xy - target_position

                position_final = optimization.root_finding.vector.secant_2d(
                    func=position_error,
                    root_guess=rays_input.position.xy,
                    step_size=0.1 * u.mm,
                    max_abs_error=1 * u.nm,
                    max_iterations=100,
                )
                rays_input.position = position_final.to_3d()
                rays_input.input_grid.pupil = grid_surf.pupil

                s1 = [slice(None)] * rays_input.position.ndim
                s2 = [slice(None)] * rays_input.position.ndim
                s3 = [slice(None)] * rays_input.position.ndim
                s4 = [slice(None)] * rays_input.position.ndim
                s1[rays_input.axis.pupil_x] = slice(None, ~0)
                s1[rays_input.axis.pupil_y] = slice(None, ~0)
                s2[rays_input.axis.pupil_x] = slice(1, None)
                s2[rays_input.axis.pupil_y] = slice(None, ~0)
                s3[rays_input.axis.pupil_x] = slice(1, None)
                s3[rays_input.axis.pupil_y] = slice(1, None)
                s4[rays_input.axis.pupil_x] = slice(None, ~0)
                s4[rays_input.axis.pupil_y] = slice(1, None)
                p1 = rays_input.position[s1]
                p2 = rays_input.position[s2]
                p3 = rays_input.position[s3]
                p4 = rays_input.position[s4]
                v31 = p1 - p3
                v42 = p2 - p4
                area = (v31.cross(v42)).length / 2
                area = area.to(u.cm ** 2)

                sh = [1, ] * rays_input.position.ndim
                sh[rays_input.axis.pupil_x] = 2
                sh[rays_input.axis.pupil_y] = 2
                kernel = np.ones(sh)
                kernel = kernel / kernel.sum()

                pad_width = [(0, 0)] * area.ndim
                pad_width[rays_input.axis.pupil_x] = (1, 1)
                pad_width[rays_input.axis.pupil_y] = (1, 1)

                if area.size > kernel.size:
                    area = np.pad(area, pad_width=pad_width, mode='edge')
                    rays_input.intensity = scipy.signal.convolve(area, kernel, mode='valid') * area.unit
                else:
                    rays_input.intensity = area

                # subtent = rays_input.input_grid.field.step_size
                # subtent = (subtent.x * subtent.y).to(u.sr)

            if surf.is_stop:
                return rays_input

        raise self.error_no_stop

    def _calc_rays_input_direction(self, rays_input: rays.Rays) -> rays.Rays:
        rays_input = rays_input.copy()
        surfaces_all_global = self.surfaces_all.flat_global
        # rays_input.transform = self.transform + self.object_surface.transform
        rays_input.transform = surfaces_all_global[0].transform
        # for surf_index, surf in enumerate(self.surfaces_all.flat_global_iter):
        for surf_index, surf in enumerate(surfaces_all_global):
            if surf.is_stop or surf.is_stop_test:
                grid_surf = self._calc_grid_rays(surf)
                target_position = grid_surf.points_pupil

                def position_error(angles: vector.Vector2D) -> vector.Vector2D:
                    rays_in = rays_input.view()
                    direction = transform.rigid.TiltX(angles.y)(vector.z_hat)
                    direction = transform.rigid.TiltY(angles.x)(direction)
                    rays_in.direction = direction
                    rays_in.input_grid.pupil = grid_surf.pupil
                    raytrace = self.surfaces_all.raytrace(rays_in, surface_last=surf)
                    return raytrace[~0].position.xy - target_position

                angles_final = optimization.root_finding.vector.secant_2d(
                    func=position_error,
                    root_guess=np.arcsin(rays_input.direction.xy),
                    step_size=1e-10 * u.deg,
                    max_abs_error=1 * u.nm,
                    max_iterations=100,
                )
                direction_final = transform.rigid.TiltX(angles_final.y)(vector.z_hat)
                direction_final = transform.rigid.TiltY(angles_final.x)(direction_final)
                rays_input.direction = direction_final
                rays_input.input_grid.pupil = grid_surf.pupil

            if surf.is_stop:
                return rays_input

        raise self.error_no_stop

    def _calc_rays_input(
            self,
            grid_rays: rays.RayGrid,
    ) -> rays.Rays:

        if grid_rays.field.points.x.unit.is_equivalent(u.rad):
            rays_input = rays.Rays.from_field_angles(
                input_grid=grid_rays,
                position=vector.Vector3D.spatial(),
            )
            rays_input.distortion_polynomial_degree = self.distortion_polynomial_degree
            rays_input.vignetting_polynomial_degree = self.vignetting_polynomial_degree
            return self._calc_rays_input_position(rays_input=rays_input)

        elif grid_rays.field.points.x.unit.is_equivalent(u.mm):
            rays_input = rays.Rays.from_field_positions(
                input_grid=grid_rays,
                direction=vector.z_hat,
            )
            rays_input.distortion_polynomial_degree = self.distortion_polynomial_degree
            rays_input.vignetting_polynomial_degree = self.vignetting_polynomial_degree
            return self._calc_rays_input_direction(rays_input=rays_input)

        # rays_input = None
        #
        # for surf_index, surf in enumerate(self.surfaces_all.flat_global_iter):
        #
        #     if not surf.is_stop and not surf.is_stop_test:
        #         continue
        #
        #     if self.field_min.quantity.unit.is_equivalent(u.rad):
        #
        #         if rays_input is None:
        #             position_guess = vector.Vector2D.spatial()
        #         else:
        #             position_guess = rays_input.position.xy
        #
        #         step_size = .1 * u.mm
        #
        #         px, py = self.pupil_x(surf), self.pupil_y(surf)
        #         # px = np.expand_dims(px, rays.Rays.axis.perp_axes(rays.Rays.axis.pupil_x))
        #         # py = np.expand_dims(py, rays.Rays.axis.perp_axes(rays.Rays.axis.pupil_y))
        #         target_position = vector.Vector2D(
        #             x=np.expand_dims(px, rays.Rays.axis.perp_axes(rays.Rays.axis.pupil_x)),
        #             y=np.expand_dims(py, rays.Rays.axis.perp_axes(rays.Rays.axis.pupil_y)),
        #         )
        #
        #         def position_error(pos: vector.Vector2D) -> vector.Vector2D:
        #             rays_in = rays.Rays.from_field_angles(
        #                 wavelength_grid=self.wavelength,
        #                 position=pos.to_3d(),
        #                 field_grid_x=self.field_x,
        #                 field_grid_y=self.field_y,
        #                 pupil_grid_x=px,
        #                 pupil_grid_y=py
        #             )
        #             rays_in.transform = self.object_surface.transform
        #             raytrace = self.surfaces_all.raytrace(rays_in, surface_last=surf)
        #
        #             return raytrace[~0].position.xy - target_position
        #
        #         position_final = optimization.root_finding.vector.secant_2d(
        #             func=position_error,
        #             root_guess=position_guess,
        #             step_size=step_size,
        #             max_abs_error=1 * u.nm,
        #             max_iterations=100,
        #         )
        #         rays_input = rays.Rays.from_field_angles(
        #             wavelength_grid=self.wavelength,
        #             position=position_final.to_3d(),
        #             field_grid_x=self.field_x,
        #             field_grid_y=self.field_y,
        #             pupil_grid_x=px,
        #             pupil_grid_y=py
        #         )
        #         rays_input.transform = self.object_surface.transform
        #
        #     else:
        #         if rays_input is None:
        #             direction_guess = vector.Vector2D(x=0 * u.deg, y=0 * u.deg)
        #         else:
        #             direction_guess = np.arcsin(rays_input.direction.xy)
        #
        #         step_size = 1e-10 * u.deg
        #
        #         px, py = self.pupil_x(surf), self.pupil_y(surf)
        #         # px = np.expand_dims(px, rays.Rays.axis.perp_axes(rays.Rays.axis.pupil_x))
        #         # py = np.expand_dims(py, rays.Rays.axis.perp_axes(rays.Rays.axis.pupil_y))
        #         target_position = vector.Vector2D(
        #             x=np.expand_dims(px, rays.Rays.axis.perp_axes(rays.Rays.axis.pupil_x)),
        #             y=np.expand_dims(py, rays.Rays.axis.perp_axes(rays.Rays.axis.pupil_y)),
        #         )
        #
        #         def position_error(angles: vector.Vector2D) -> vector.Vector2D:
        #             direction = transform.rigid.TiltX(angles.y)(vector.z_hat)
        #             direction = transform.rigid.TiltY(angles.x)(direction)
        #             rays_in = rays.Rays.from_field_positions(
        #                 wavelength_grid=self.wavelength,
        #                 direction=direction,
        #                 field_grid_x=self.field_x,
        #                 field_grid_y=self.field_y,
        #                 pupil_grid_x=px,
        #                 pupil_grid_y=py,
        #             )
        #             rays_in.transform = self.object_surface.transform
        #             raytrace = self.surfaces_all.raytrace(rays_in, surface_last=surf)
        #
        #             return raytrace[~0].position.xy - target_position
        #
        #         angles_final = optimization.root_finding.vector.secant_2d(
        #             func=position_error,
        #             root_guess=direction_guess,
        #             step_size=step_size,
        #             max_abs_error=1 * u.nm,
        #             max_iterations=100,
        #         )
        #         direction_final = transform.rigid.TiltX(angles_final.y)(vector.z_hat)
        #         direction_final = transform.rigid.TiltY(angles_final.x)(direction_final)
        #         rays_input = rays.Rays.from_field_positions(
        #             wavelength_grid=self.wavelength,
        #             direction=direction_final,
        #             field_grid_x=self.field_x,
        #             field_grid_y=self.field_y,
        #             pupil_grid_x=px,
        #             pupil_grid_y=py,
        #         )
        #         rays_input.transform = self.object_surface.transform
        #
        #     if surf.is_stop:
        #         return rays_input
        #
        # raise ValueError('No stop defined')

    def psf(
            self,
            bins: typ.Union[int, typ.Tuple[int, int]] = 10,
            limit_min: typ.Optional[vector.Vector2D] = None,
            limit_max: typ.Optional[vector.Vector2D] = None,
            use_vignetted: bool = False,
            relative_to_centroid: typ.Tuple[bool, bool] = (False, False),
    ) -> typ.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.rays_output.pupil_hist2d(
            bins=bins,
            limit_min=limit_min,
            limit_max=limit_max,
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
        if np.array(self.wavelength != other.wavelength).any():
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
            surf.plot(ax=ax, plot_annotations=False)

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
            ax: matplotlib.axes.Axes,
            components: typ.Tuple[str, str] = ('x', 'y'),
            component_z: typ.Optional[str] = None,
            plot_kwargs: typ.Optional[typ.Dict[str, typ.Any]] = None,
            # color: typ.Optional[str] = None,
            # linewidth: typ.Optional[float] = None,
            # linestyle: typ.Optional[str] = None,
            transform_extra: typ.Optional[transform.rigid.TransformList] = None,
            surface_first: typ.Optional[surface.Surface] = None,
            surface_last: typ.Optional[surface.Surface] = None,
            plot_rays: bool = True,
            color_axis: int = rays.Rays.axis.wavelength,
            plot_vignetted: bool = False,
            plot_colorbar: bool = True,
            plot_baffles: bool = True,
            plot_breadboard: bool = True,
            plot_annotations: bool = True,
            annotation_text_y: float = 1.05,
    ) -> typ.Tuple[typ.List[matplotlib.lines.Line2D], typ.Optional[matplotlib.colorbar.Colorbar]]:

        if plot_kwargs is not None:
            plot_kwargs = {**self.plot_kwargs, **plot_kwargs}
        else:
            plot_kwargs = self.plot_kwargs

        # if color is None:
        #     color = self.color
        # if linewidth is None:
        #     linewidth = self.linewidth
        # if linestyle is None:
        #     linestyle = self.linestyle

        surfaces = self.surfaces_all.flat_local

        if transform_extra is None:
            transform_extra = transform.rigid.TransformList()
        # transform_extra = transform_extra + self.transform_all

        if surface_first is None:
            surface_first = surfaces[0]
        if surface_last is None:
            surface_last = surfaces[~0]
        surface_index_first = surfaces.index(surface_first)
        surface_index_last = surfaces.index(surface_last)

        surf_slice = slice(surface_index_first, surface_index_last + 1)

        lines = []

        colorbar = None
        if plot_rays:
            raytrace_slice = self.raytrace[surf_slice]  # type: rays.RaysList
            rlines, colorbar = raytrace_slice.plot(
                ax=ax,
                components=components,
                component_z=component_z,
                plot_kwargs=plot_kwargs,
                transform_extra=transform_extra,
                color_axis=color_axis,
                plot_vignetted=plot_vignetted,
                plot_colorbar=plot_colorbar,
            )
            lines += rlines

        surfaces_slice = self.surfaces_all.flat_global[surf_slice]  # type: surfaces.SurfaceList
        lines += surfaces_slice.plot(
            ax=ax,
            components=components,
            component_z=component_z,
            plot_kwargs=plot_kwargs,
            # color=color,
            # linewidth=linewidth,
            # linestyle=linestyle,
            transform_extra=transform_extra,
            to_global=True,
            plot_annotations=plot_annotations,
            annotation_text_y=annotation_text_y,
        )

        if plot_baffles:
            if self.baffles is not None:
                self.baffles.plot(
                    ax=ax,
                    components=components,
                    plot_kwargs=plot_kwargs,
                    transform_extra=transform_extra,
                )

        if plot_breadboard:
            if self.breadboard is not None:
                self.breadboard.plot(
                    ax=ax,
                    components=components,
                    transform_extra=transform_extra + self.transform_all,
                    to_global=True,
                )

        return lines, colorbar

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
        other.wavelength = self.wavelength
        other.pupil_samples = self.pupil_samples
        other.pupil_margin = self.pupil_margin
        other.field_samples = self.field_samples
        other.field_margin = self.field_margin
        other.pointing = self.pointing
        other.roll = self.roll
        other.baffles_blank = self.baffles_blank
        other.baffles_hull_axes = self.baffles_hull_axes
        other.breadboard = self.breadboard
        other.distortion_polynomial_degree = self.distortion_polynomial_degree
        other.vignetting_polynomial_degree = self.vignetting_polynomial_degree
        return other

    def copy(self) -> 'System':
        other = super().copy()  # type: System
        other.object_surface = self.object_surface.copy()
        other.surfaces = self.surfaces.copy()
        other.wavelength = self.wavelength.copy()
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
        other.distortion_polynomial_degree = self.distortion_polynomial_degree
        other.vignetting_polynomial_degree = self.vignetting_polynomial_degree
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
            plot_colorbar: bool = True,
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
                plot_baffles=plot_baffles,
                plot_colorbar=plot_colorbar,
            )

        if plot_baffles:
            self.baffles.plot(ax=ax, components=components, transform_extra=transform_extra)

        return ax
