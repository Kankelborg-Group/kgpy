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
from ezdxf.addons.r12writer import R12FastStreamWriter
# from kgpy import mixin, linspace, vector, optimization, transform, obs, grid
from kgpy import mixin
from kgpy import labeled
import kgpy.vectors
from kgpy import transforms
from kgpy import optimization
from kgpy.io import dxf
from . import aberrations
from . import vectors
from . import rays
from .surfaces import Surface, SurfaceList
from .baffles import Baffle, BaffleList
from .breadboards import Breadboard

__all__ = [
    'aberrations',
    'rays',
    'surfaces',
    'components',
    'baffles',
    'breadboards',
    'System',
    'SystemList',
]

SystemT = typ.TypeVar('SystemT', bound='System')


@dataclasses.dataclass
class System(
    dxf.WritableMixin,
    transforms.Transformable,
    mixin.Plottable,
    mixin.Broadcastable,
    mixin.Named,
):
    """
    Model of an optical system.
    """
    #: Surface representing the light source
    transform_pointing: transforms.TransformList = dataclasses.field(default_factory=transforms.TransformList)
    object_surface: Surface = dataclasses.field(default_factory=Surface)
    surfaces: SurfaceList = dataclasses.field(default_factory=SurfaceList)
    object_grid_normalized: vectors.ObjectVector = dataclasses.field(default_factory=vectors.ObjectVector)
    field_margin: u.Quantity = 1 * u.nm  #: Margin between edge of field and nearest ray
    pupil_margin: u.Quantity = 1 * u.nm  #: Margin between edge of pupil and nearest ray
    # pointing: vector.Cartesian2D = dataclasses.field(default_factory=lambda: vector.Cartesian2D() * u.deg)
    # roll: u.Quantity = 0 * u.deg
    baffles_blank: BaffleList = dataclasses.field(default_factory=BaffleList)
    baffles_hull_axes: typ.Optional[typ.Tuple[int, ...]] = None
    breadboard: typ.Optional[Breadboard] = None
    # tolerance_axes: typ.Dict[str, int] = dataclasses.field(default_factory=lambda: {})
    # focus_axes: typ.Dict[str, int] = dataclasses.field(default_factory=lambda: {})
    distortion_polynomial_degree: int = 2
    vignetting_polynomial_degree: int = 1

    def __post_init__(self: SystemT):
        self.update()

    def update(self: SystemT) -> None:
        self._rays_input_cache = None
        self._rays_input_resample_entrance_cache = None
        self._raytrace_cache = None
        self._raytrace_resample_entrance_cache = None
        self._baffles_cache = None

    @property
    def transform_all(self) -> transforms.TransformList:
        return self.transform_pointing + self.transform

    @property
    def surfaces_all(self) -> SurfaceList:
        surfaces = self.surfaces.copy()
        surfaces.transform = self.transform_all
        return SurfaceList([self.object_surface, surfaces])

    @property
    def field_stop(self: SystemT) -> Surface:
        for surf in self.surfaces_all.flat_global_iter:
            if surf.is_field_stop:
                return surf
        raise self.error_no_pupil_stop

    @property
    def pupil_stop(self: SystemT) -> Surface:
        for surf in self.surfaces_all.flat_global_iter:
            if surf.is_pupil_stop:
                return surf
        raise self.error_no_pupil_stop

    @property
    def wavelength(self) -> u.Quantity:
        return self.rays_input.input.wavelength

    # def _calc_grid_rays(self, surf: surface.Surface) -> rays.RayGrid:
    #     grid_rays = self.grid_rays.copy()
    #     grid_rays.field.min = self.object_surface.aperture.min.xy + self.field_margin
    #     grid_rays.field.max = self.object_surface.aperture.max.xy - self.field_margin
    #     grid_rays.pupil.min = surf.aperture.min.xy + self.pupil_margin
    #     grid_rays.pupil.max = surf.aperture.max.xy - self.pupil_margin
    #     return grid_rays
    #
    # @property
    # def grid_rays_stop(self) -> rays.RayGrid:
    #     return self._calc_grid_rays(self.surface_stop)

    @property
    def baffle_lofts(self) -> typ.Dict[int, typ.Tuple[Surface, Surface]]:
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
    def baffles(self) -> BaffleList:
        if self._baffles_cache is None:
            self._baffles_cache = self.calc_baffles(self.baffles_blank)
        return self._baffles_cache

    def calc_baffles(
            self,
            baffles_blank: BaffleList,
            transform_extra: typ.Optional[transforms.TransformList] = None,
    ) -> BaffleList:

        return None

        if transform_extra is None:
            transform_extra = transforms.TransformList()

        if baffles_blank:

            baffles = baffles_blank
            baffles = baffles.concat_apertures_from_raytrace(
                raytrace=self.raytrace, transform_extra=transform_extra, hull_axes=self.baffles_hull_axes, color='red')

            baffles = baffles.concat_apertures_from_lofts(lofts=self.baffle_lofts, transform_extra=transform_extra)

        else:
            baffles = baffles_blank

        return baffles

    @property
    def raytrace(self) -> rays.RayFunctionList:
        if self._raytrace_cache is None:
            self._raytrace_cache = self.surfaces_all.raytrace(self.rays_input)
        return self._raytrace_cache

    @property
    def rays_output(self) -> rays.RayFunction:
        return self.raytrace[~0]

    @property
    def rays_input(self) -> rays.RayFunction:
        if self._rays_input_cache is None:
            self._rays_input_cache = self._calc_rays_input(self.object_grid_normalized)
        return self._rays_input_cache

    # @property
    # def rays_input_resample_entrance(self) -> rays.RayFunction:
    #     if self._rays_input_resample_entrance_cache is None:
    #         rays_input = self.rays_input.copy()
    #         rays_output = self.rays_output
    #         if rays_input.input_grid.field.points.x.unit.is_equivalent(u.rad):
    #             position = np.broadcast_to(rays_input.position, rays_output.position.shape, subok=True).copy()
    #             position[~rays_output.mask] = np.nan
    #             rays_input.input_grid.pupil.min = np.nanmin(position, axis=tuple(rays_output.axis.all))
    #             rays_input.input_grid.pupil.max = np.nanmax(position, axis=tuple(rays_output.axis.all))
    #             rays_input.position = rays_input.input_grid.points_pupil.to_3d()
    #             rays_input.intensity = rays_input.input_grid.pupil.step_size.x * rays_input.input_grid.pupil.step_size.y
    #             self._rays_input_resample_entrance_cache = rays_input
    #         else:
    #             raise NotImplementedError
    #     return self._rays_input_resample_entrance_cache
    #
    # @property
    # def raytrace_resample_entrance(self) -> rays.RayFunctionList:
    #     if self._raytrace_resample_entrance_cache is None:
    #         self._raytrace_resample_entrance_cache = self.surfaces_all.raytrace(self.rays_input_resample_entrance)
    #     return self._raytrace_resample_entrance_cache
    #
    # @property
    # def rays_output_resample_entrance(self) -> rays.RayFunction:
    #     return self.raytrace_resample_entrance[~0]

    # error_no_pupil_stop = ValueError('no pupil stop defined')
    # error_no_field_stop = ValueError('no field stop defined')
    #
    # def _calc_rays_input_position(self, rays_input: rays.RayFunction) -> rays.RayFunction:
    #     rays_input = rays_input.copy()
    #     surfaces_all_global = self.surfaces_all.flat_global
    #     # rays_input.transform = self.transform + self.object_surface.transform
    #     rays_input.transform = surfaces_all_global[0].transform
    #     # for surf_index, surf in enumerate(self.surfaces_all.flat_global_iter):
    #     for surf_index, surf in enumerate(surfaces_all_global):
    #         if surf.is_stop or surf.is_stop_test:
    #             grid_surf = self._calc_grid_rays(surf)
    #             target_position = grid_surf.points_pupil
    #
    #             def position_error(pos: vector.Vector2D) -> vector.Vector2D:
    #                 rays_in = rays_input.copy_shallow()
    #                 rays_in.position = pos.to_3d()
    #                 rays_in.input_grid.pupil = grid_surf.pupil
    #                 raytrace = self.surfaces_all.raytrace(rays_in, surface_last=surf)
    #                 return raytrace[~0].position.xy - target_position
    #
    #             position_final = optimization.root_finding.vector.secant_2d(
    #                 func=position_error,
    #                 root_guess=rays_input.position.xy,
    #                 step_size=0.1 * u.mm,
    #                 max_abs_error=1 * u.nm,
    #                 max_iterations=100,
    #             )
    #             rays_input.position = position_final.to_3d()
    #             rays_input.input_grid.pupil = grid_surf.pupil
    #
    #             s1 = [slice(None)] * rays_input.position.ndim
    #             s2 = [slice(None)] * rays_input.position.ndim
    #             s3 = [slice(None)] * rays_input.position.ndim
    #             s4 = [slice(None)] * rays_input.position.ndim
    #             s1[rays_input.axis.pupil_x] = slice(None, ~0)
    #             s1[rays_input.axis.pupil_y] = slice(None, ~0)
    #             s2[rays_input.axis.pupil_x] = slice(1, None)
    #             s2[rays_input.axis.pupil_y] = slice(None, ~0)
    #             s3[rays_input.axis.pupil_x] = slice(1, None)
    #             s3[rays_input.axis.pupil_y] = slice(1, None)
    #             s4[rays_input.axis.pupil_x] = slice(None, ~0)
    #             s4[rays_input.axis.pupil_y] = slice(1, None)
    #             p1 = rays_input.position[s1]
    #             p2 = rays_input.position[s2]
    #             p3 = rays_input.position[s3]
    #             p4 = rays_input.position[s4]
    #             v31 = p1 - p3
    #             v42 = p2 - p4
    #             area = (v31.cross(v42)).length / 2
    #             area = area.to(u.cm ** 2)
    #
    #             sh = [1, ] * rays_input.position.ndim
    #             sh[rays_input.axis.pupil_x] = 2
    #             sh[rays_input.axis.pupil_y] = 2
    #             kernel = np.ones(sh)
    #             kernel = kernel / kernel.sum()
    #
    #             pad_width = [(0, 0)] * area.ndim
    #             pad_width[rays_input.axis.pupil_x] = (1, 1)
    #             pad_width[rays_input.axis.pupil_y] = (1, 1)
    #
    #             if area.size > kernel.size:
    #                 area = np.pad(area, pad_width=pad_width, mode='edge')
    #                 rays_input.intensity = scipy.signal.convolve(area, kernel, mode='valid') * area.unit
    #             else:
    #                 rays_input.intensity = area
    #
    #             # subtent = rays_input.input_grid.field.step_size
    #             # subtent = (subtent.x * subtent.y).to(u.sr)
    #
    #         if surf.is_stop:
    #             return rays_input
    #
    #     raise self.error_no_stop
    #
    # def _calc_rays_input_direction(self, rays_input: rays.RayFunction) -> rays.RayFunction:
    #     rays_input = rays_input.copy()
    #     surfaces_all_global = self.surfaces_all.flat_global
    #     # rays_input.transform = self.transform + self.object_surface.transform
    #     rays_input.transform = surfaces_all_global[0].transform
    #     # for surf_index, surf in enumerate(self.surfaces_all.flat_global_iter):
    #     for surf_index, surf in enumerate(surfaces_all_global):
    #         if surf.is_stop or surf.is_stop_test:
    #             grid_surf = self._calc_grid_rays(surf)
    #             target_position = grid_surf.points_pupil
    #
    #             def position_error(angles: vector.Vector2D) -> vector.Vector2D:
    #                 rays_in = rays_input.copy_shallow()
    #                 direction = transform.rigid.TiltX(angles.y)(vector.z_hat)
    #                 direction = transform.rigid.TiltY(angles.x)(direction)
    #                 rays_in.direction = direction
    #                 rays_in.input_grid.pupil = grid_surf.pupil
    #                 raytrace = self.surfaces_all.raytrace(rays_in, surface_last=surf)
    #                 return raytrace[~0].position.xy - target_position
    #
    #             angles_final = optimization.root_finding.vector.secant_2d(
    #                 func=position_error,
    #                 root_guess=np.arcsin(rays_input.direction.xy),
    #                 step_size=1e-10 * u.deg,
    #                 max_abs_error=1 * u.nm,
    #                 max_iterations=100,
    #             )
    #             direction_final = transform.rigid.TiltX(angles_final.y)(vector.z_hat)
    #             direction_final = transform.rigid.TiltY(angles_final.x)(direction_final)
    #             rays_input.direction = direction_final
    #             rays_input.input_grid.pupil = grid_surf.pupil
    #
    #         if surf.is_stop:
    #             return rays_input
    #
    #     raise self.error_no_stop

    # @classmethod
    # def _angle_to_direction(cls: typ.Type[SystemT], angle: kgpy.vector.Cartesian2D) -> kgpy.vector.Cartesian3D:
    #     transform = kgpy.transforms.TransformList([
    #         kgpy.transforms.RotationY(-angle.x),
    #         kgpy.transforms.RotationX(angle.y),
    #     ])
    #     return transform(kgpy.vector.Cartesian3D.z_hat(), translate=False)
    #
    # @classmethod
    # def _direction_to_angle(cls: typ.Type[SystemT], direction: kgpy.vector.Cartesian3D) -> kgpy.vector.Cartesian2D:
    #     return kgpy.vector.Cartesian2D(
    #         x=-np.arctan2(direction.x, direction.z),
    #         y=np.arcsin(direction.y / direction.length),
    #     )

    def _calc_rays_input_stops_only(self: SystemT, object_grid_normalized: vectors.ObjectVector) -> rays.RayFunction:
        surfaces_all_global = self.surfaces_all.flat_global

        result = rays.RayFunction()
        result.input = object_grid_normalized.copy_shallow()
        result.output.wavelength = result.input.wavelength

        visited_pupil_stops = []
        visited_pupil_stop = False
        while not visited_pupil_stop:

            surfaces_subsystem = SurfaceList()
            for surf in surfaces_all_global:

                if surfaces_subsystem:
                    surfaces_subsystem.append(surf)

                if surf.is_field_stop or surf.is_pupil_stop or surf.is_pupil_stop_test:
                    if surf.is_field_stop:
                        result.input.field_x.start = surf.aperture.min.x + self.field_margin
                        result.input.field_y.start = surf.aperture.min.y + self.field_margin
                        result.input.field_x.stop = surf.aperture.max.x - self.field_margin
                        result.input.field_y.stop = surf.aperture.max.y - self.field_margin

                    if surf in visited_pupil_stops:
                        continue

                    if surf.is_pupil_stop or surf.is_pupil_stop_test:
                        result.input.pupil_x.start = surf.aperture.min.x + self.pupil_margin
                        result.input.pupil_y.start = surf.aperture.min.y + self.pupil_margin
                        result.input.pupil_x.stop = surf.aperture.max.x - self.pupil_margin
                        result.input.pupil_y.stop = surf.aperture.max.y - self.pupil_margin
                        visited_pupil_stops.append(surf)

                    if surf.is_pupil_stop:
                        visited_pupil_stop = True

                    if not surfaces_subsystem:
                        surfaces_subsystem.append(surf)
                    else:
                        break

            result.output.transform = surfaces_subsystem[0].transform

            if surfaces_subsystem[~0].is_field_stop:
                position_target = result.input.field_xy
            else:
                position_target = result.input.pupil_xy

            unit_aperture_first = surfaces_subsystem[0].aperture.wire.unit
            if unit_aperture_first.is_equivalent(u.mm):

                if surfaces_subsystem[0].is_field_stop:
                    result.output.position = result.input.field_xy.to_3d()
                else:
                    result.output.position = result.input.pupil_xy.to_3d()
                if surfaces_subsystem[0].sag is not None:
                    result.output.position.z = surfaces_subsystem[0].sag(result.output.position)

                def position_error(angles: kgpy.vectors.Cartesian2D) -> kgpy.vectors.Cartesian2D:
                    rays_in = result.copy_shallow()
                    rays_in.output = rays_in.output.copy_shallow()
                    rays_in.output.angles = angles
                    raytrace_test = surfaces_subsystem[1:].raytrace(rays_in)
                    return raytrace_test[~0].output.position.xy - position_target

                result.output.angles = optimization.root_finding.secant(
                    func=position_error,
                    root_guess=result.output.angles,
                    step_size=kgpy.vectors.Cartesian2D() + 1e-10 * u.deg,
                    max_abs_error=1 * u.nm,
                )

            elif unit_aperture_first.is_equivalent(u.deg):

                if surfaces_subsystem[0].is_field_stop:
                    result.output.angles = result.input.field_xy
                else:
                    result.output.angles = result.input.pupil_xy

                def position_error(pos: kgpy.vectors.Cartesian2D) -> kgpy.vectors.Cartesian2D:
                    rays_in = result.copy_shallow()
                    rays_in.output = rays_in.output.copy_shallow()
                    rays_in.output.position = pos.to_3d()
                    if surfaces_subsystem[0].sag is not None:
                        rays_in.output.position.z = surfaces_subsystem[0].sag(rays_in.output.position)
                    raytrace_test = surfaces_subsystem[1:].raytrace(rays_in)
                    return raytrace_test[~0].output.position.xy - position_target

                position_final = optimization.root_finding.secant(
                    func=position_error,
                    root_guess=result.output.position.xy,
                    step_size=kgpy.vectors.Cartesian2D() + 0.1 * u.mm,
                    max_abs_error=1 * u.nm,
                )
                result.output.position = position_final.to_3d()
                if surfaces_subsystem[0].sag is not None:
                    result.output.position.z = surfaces_subsystem[0].sag(result.output.position)

            else:
                raise NotImplementedError

        return result

    def _calc_rays_input_stops(self: SystemT, object_grid_normalized: vectors.ObjectVector) -> rays.RayFunction:

        surfaces_all_global = self.surfaces_all.flat_global

        surfaces_subsystem = SurfaceList()
        for surf in surfaces_all_global:
            surfaces_subsystem.append(surf)
            if surf.is_field_stop or surf.is_pupil_stop or surf.is_pupil_stop_test:
                break

        surfaces_subsystem.reverse()

        rays_stop = self._calc_rays_input_stops_only(object_grid_normalized)

        raytrace = surfaces_subsystem.raytrace(rays_stop)

        return raytrace[~0]

    def _calc_rays_input(self: SystemT, object_grid_normalized: vectors.ObjectVector) -> rays.RayFunction:

        rays_input_stops = self._calc_rays_input_stops(object_grid_normalized)

        if self.object_surface.aperture is not None:
            aperture_unit = self.object_surface.aperture.wire.unit
            if aperture_unit.is_equivalent(u.deg):
                object_at_infinity = True
            elif aperture_unit.is_equivalent(u.mm):
                object_at_infinity = False
            else:
                raise ValueError
        else:
            object_at_infinity = True

        result = rays.RayFunction()
        result.input = object_grid_normalized.copy_shallow()
        result.output.transform = self.object_surface.transform
        result.output.wavelength = result.input.wavelength

        axes = [
            'wavelength',
            'velocity_los',
            'field_x',
            'field_y',
            'pupil_x',
            'pupil_y',
            '_distribution',
        ]

        position = rays_input_stops.output.position
        angle = rays_input_stops.output.angles
        # dummy = labeled.Array.empty({a: 1 for a in axes})
        # # shape = dummy.broadcast_shapes(position, angle)
        # # print(shape)
        # position = position.broadcast_to(dummy.shape_broadcasted(position))
        # angle = angle.broadcast_to(dummy.shape_broadcasted(angle))

        if object_at_infinity:
            result.input.field_x.start = angle.x.array_labeled.min(axis=axes, where=where, initial=np.inf)
            result.input.field_y.start = angle.y.array_labeled.min(axis=axes, where=where, initial=np.inf)
            result.input.field_x.stop = angle.x.array_labeled.max(axis=axes, where=where, initial=-np.inf)
            result.input.field_y.stop = angle.y.array_labeled.max(axis=axes, where=where, initial=-np.inf)
            result.input.pupil_x.start = position.x.array_labeled.min(axis=axes, where=where, initial=np.inf)
            result.input.pupil_y.start = position.y.array_labeled.min(axis=axes, where=where, initial=np.inf)
            result.input.pupil_x.stop = position.x.array_labeled.max(axis=axes, where=where, initial=-np.inf)
            result.input.pupil_y.stop = position.y.array_labeled.max(axis=axes, where=where, initial=-np.inf)

            result.output.position = result.input.pupil_xy.to_3d()
            result.output.angles = result.input.field_xy

        else:
            result.input.pupil_x.start = angle.x.array_labeled.min(axis=axes, where=where, initial=np.inf)
            result.input.pupil_y.start = angle.y.array_labeled.min(axis=axes, where=where, initial=np.inf)
            result.input.pupil_x.stop = angle.x.array_labeled.max(axis=axes, where=where, initial=-np.inf)
            result.input.pupil_y.stop = angle.y.array_labeled.max(axis=axes, where=where, initial=-np.inf)
            result.input.field_x.start = position.x.array_labeled.min(axis=axes, where=where, initial=np.inf)
            result.input.field_y.start = position.y.array_labeled.min(axis=axes, where=where, initial=np.inf)
            result.input.field_x.stop = position.x.array_labeled.max(axis=axes, where=where, initial=-np.inf)
            result.input.field_y.stop = position.y.array_labeled.max(axis=axes, where=where, initial=-np.inf)

            result.output.position = result.input.field_xy.to_3d()
            result.output.angles = result.input.pupil_xy

        return result











    # def _calc_rays_input(
    #         self,
    #         grid_rays: rays.RayGrid,
    # ) -> rays.RayFunction:
    #
    #     if grid_rays.field.points.x.unit.is_equivalent(u.rad):
    #         rays_input = rays.Rays.from_field_angles(
    #             input_grid=grid_rays,
    #             position=vector.Vector3D.spatial(),
    #         )
    #         rays_input.distortion_polynomial_degree = self.distortion_polynomial_degree
    #         rays_input.vignetting_polynomial_degree = self.vignetting_polynomial_degree
    #         return self._calc_rays_input_position(rays_input=rays_input)
    #
    #     elif grid_rays.field.points.x.unit.is_equivalent(u.mm):
    #         rays_input = rays.Rays.from_field_positions(
    #             input_grid=grid_rays,
    #             direction=vector.z_hat,
    #         )
    #         rays_input.distortion_polynomial_degree = self.distortion_polynomial_degree
    #         rays_input.vignetting_polynomial_degree = self.vignetting_polynomial_degree
    #         return self._calc_rays_input_direction(rays_input=rays_input)

    @property
    def psf_diffraction(self) -> aberrations.PointSpread:
        rays_output = self.rays_output_resample_entrance
        intensity = rays_output.intensity.copy()
        intensity[~rays_output.mask] = 0
        intensity = intensity.mean(rays_output.axis.velocity_los)
        return aberration.psf.DiscretePSF.from_pupil_function(
            pupil_function=intensity,
            grid=aberration.psf.Grid(
                field=rays_output.input_grid.field,
                position=rays_output.input_grid.pupil,
                wavelength=rays_output.input_grid.wavelength,
            ),
        )


    # def psf(
    #         self,
    #         bins: typ.Union[int, typ.Tuple[int, int]] = 10,
    #         limit_min: typ.Optional[vector.Vector2D] = None,
    #         limit_max: typ.Optional[vector.Vector2D] = None,
    #         use_vignetted: bool = False,
    #         relative_to_centroid: typ.Tuple[bool, bool] = (False, False),
    # ) -> typ.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     return self.rays_output.pupil_hist2d(
    #         bins=bins,
    #         limit_min=limit_min,
    #         limit_max=limit_max,
    #         use_vignetted=use_vignetted,
    #         relative_to_centroid=relative_to_centroid,
    #     )

    def generic_fit(
            self,
            observed_images: u.Quantity,
            target_images: u.Quantity,
            target_images_min: kgpy.vectors.Cartesian2D,
            target_images_max: kgpy.vectors.Cartesian2D,
            factory: typ.Callable[[typ.List[u.Quantity], 'System', int], 'System'],
            # channel_index: typ.Union[int, typ.Tuple[int, ...]] = (),
            params_guess: typ.Optional[typ.List[u.Quantity]] = None,
            params_min: typ.Optional[typ.List[u.Quantity]] = None,
            params_max: typ.Optional[typ.List[u.Quantity]] = None,
            use_correlate: bool = False,
            x_axis: int = ~2,
            y_axis: int = ~1,
            w_axis: int = ~0,
    ) -> 'System':

        axes_all = x_axis, y_axis, w_axis

        observed_images = np.expand_dims(observed_images, w_axis)
        observed_images_shape = list(observed_images.shape)
        observed_images_shape[w_axis] = target_images.shape[w_axis]
        observed_images = observed_images.reshape(observed_images_shape)

        observed_images = observed_images / np.median(observed_images, axis=axes_all, keepdims=True)
        target_images = target_images / np.median(target_images, axis=axes_all, keepdims=True)

        if params_guess is not None:
            params_unit = [q.unit for q in params_guess]
        else:
            params_unit = [q.unit for q in params_min]

        def factory_value(params: np.ndarray, other: 'System', chan_index: int) -> 'System':
            params = [param * unit for param, unit in zip(params, params_unit)]
            return factory(params, other, chan_index)

        def objective(params: np.ndarray, chan_index: int) -> float:
            other = factory_value(params=params, chan_index=chan_index)
            test_images = other(
                data=observed_images,
                wavelength=other.wavelength,
                spatial_input_min=kgpy.vectors.Vector2D(x=0 * u.pix, y=0 * u.pix),
                spatial_input_max=kgpy.vectors.Vector2D(
                    x=observed_images.shape[x_axis],
                    y=observed_images.shape[y_axis],
                ),
                spatial_output_min=target_images_min,
                spatial_output_max=target_images_max,
                spatial_samples_output=kgpy.vectors.Vector2D(
                    x=target_images.shape[x_axis],
                    y=target_images.shape[y_axis],
                ),
                inverse=True,
            )
            if use_correlate:
                corr = scipy.signal.correlate(
                    in1=np.nan_to_num(test_images[chan_index]),
                    in2=target_images,
                    mode='same',
                )
                corr = np.prod(corr, axis=w_axis)
                lag = np.array(np.unravel_index(np.argmax(corr), corr.shape)) - np.array(corr.shape) // 2
                test_images[chan_index] = np.roll(test_images[chan_index], -lag, axis=(x_axis, y_axis))

            diff = test_images[chan_index] - target_images
            norm = np.sqrt(np.mean(np.square(diff)))
            return norm

        other = self.copy_shallow()
        shape = observed_images.shape[:~2]
        for i in range(np.prod(shape)):
            index = np.unravel_index(i, shape)

            params_converted_min = [param[index].to(unit).value for param, unit in zip(params_min, params_unit)]
            params_converted_max = [param[index].to(unit).value for param, unit in zip(params_max, params_unit)]

            params_converted_min = np.array(params_converted_min)
            params_converted_max = np.array(params_converted_max)

            x0 = scipy.optimize.brute(
                func=objective,
                ranges=np.stack([params_converted_min, params_converted_max], axis=~0),
                args=(other, index, ),
                disp=True,
            )

            other = factory_value(params=x0, other=other, chan_index=index)

        return other



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
            surf: typ.Optional[Surface] = None,
            color_axis: str = 'wavelength',
            plot_apertures: bool = True,
            plot_vignetted: bool = False,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()

        surfaces = self.surfaces_all.flat_local

        if surf is None:
            surf = surfaces[~0]

        surf_index = surfaces.index(surf)
        surf_rays = self.raytrace[surf_index].copy_shallow()
        surf_rays.vignetted_mask = self.rays_output.vignetted_mask

        surf_rays.plot_position(ax=ax, color_axis=color_axis, plot_vignetted=plot_vignetted)

        if plot_apertures:
            surf.plot(ax=ax, plot_annotations=False)

        return ax

    def plot_projections(
            self,
            surface_first: typ.Optional[Surface] = None,
            surface_last: typ.Optional[Surface] = None,
            color_axis: str = 'wavelength',
            plot_vignetted: bool = False,
            plot_rays: bool = True,
    ) -> plt.Figure:
        fig, axs = plt.subplots(2, 2, sharex='col', sharey='row')

        xy = 0, 0
        yz = 0, 1
        xz = 1, 1

        axs[xy].invert_xaxis()

        ax_indices = [xy, yz, xz]
        components_x = ['x', 'z', 'z']
        components_y = ['y', 'y', 'x']
        # planes = [
        #     ('x', 'y'),
        #     ('z', 'y'),
        #     ('z', 'x'),
        # ]
        for ax_index, component_x, component_y in zip(ax_indices, components_x, components_y):
            self.plot(
                ax=axs[ax_index],
                component_x=component_x,
                component_y=component_y,
                surface_first=surface_first,
                surface_last=surface_last,
                color_axis=color_axis,
                plot_vignetted=plot_vignetted,
                plot_rays=plot_rays,
            )
        #     if plot_rays:
        #         axs[ax_index].get_legend().remove()
        #
        # handles, labels = axs[xy].get_legend_handles_labels()
        # label_dict = dict(zip(labels, handles))
        # fig.legend(label_dict.values(), label_dict.keys(), loc='top left', bbox_to_anchor=(1.0, 1.0))

        return fig

    def plot(
            self,
            ax: matplotlib.axes.Axes,
            component_x: str = 'z',
            component_y: str = 'x',
            component_z: str = 'y',
            transform_extra: typ.Optional[transforms.TransformList] = None,
            surface_first: typ.Optional[Surface] = None,
            surface_last: typ.Optional[Surface] = None,
            plot_rays: bool = True,
            color_axis: str = 'wavelength',
            plot_vignetted: bool = False,
            plot_colorbar: bool = True,
            plot_baffles: bool = True,
            plot_breadboard: bool = True,
            plot_annotations: bool = True,
            annotation_text_y: float = 1.05,
            **kwargs,
    ) -> typ.Tuple[typ.List[matplotlib.lines.Line2D], typ.Optional[matplotlib.colorbar.Colorbar]]:

        kwargs = {**self.plot_kwargs, **kwargs}

        surfaces = self.surfaces_all.flat_local

        if transform_extra is None:
            transform_extra = transforms.TransformList()
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
                component_x=component_x,
                component_y=component_y,
                component_z=component_z,
                transform_extra=transform_extra,
                color_axis=color_axis,
                plot_vignetted=plot_vignetted,
                **kwargs
            )
            lines += rlines

        surfaces_slice = self.surfaces_all.flat_global[surf_slice]  # type: surfaces.SurfaceList
        lines += surfaces_slice.plot(
            ax=ax,
            component_x=component_x,
            component_y=component_y,
            component_z=component_z,
            transform_extra=transform_extra,
            to_global=True,
            plot_annotations=plot_annotations,
            annotation_text_y=annotation_text_y,
            **kwargs
        )

        if plot_baffles:
            if self.baffles is not None:
                self.baffles.plot(
                    ax=ax,
                    component_x=component_x,
                    component_y=component_y,
                    component_z=component_z,
                    transform_extra=transform_extra,
                    **kwargs,
                )

        if plot_breadboard:
            if self.breadboard is not None:
                self.breadboard.plot(
                    ax=ax,
                    component_x=component_x,
                    component_y=component_y,
                    component_z=component_z,
                    transform_extra=transform_extra + self.transform_all,
                    to_global=True,
                    **kwargs,
                )

        return lines, colorbar

    def write_to_dxf(
            self: SystemT,
            file_writer: R12FastStreamWriter,
            unit: u.Unit,
            transform_extra: typ.Optional[transforms.TransformList] = None,
    ) -> None:

        self.surfaces_all.flat_global.write_to_dxf(
            file_writer=file_writer,
            unit=unit,
            transform_extra=transform_extra,
        )

        self.raytrace.write_to_dxf(
            file_writer=file_writer,
            unit=unit,
            transform_extra=transform_extra,
        )


@dataclasses.dataclass
class SystemList(
    mixin.DataclassList[System],
):
    baffles_blank: BaffleList = dataclasses.field(default_factory=BaffleList)

    def __post_init__(self):
        self.update()

    def update(self) -> typ.NoReturn:
        self._baffles_cache = None

    @property
    def baffles(self) -> BaffleList:
        if self._baffles_cache is None:
            baffles = self.baffles_blank.copy()
            if baffles:
                for sys in self:
                    new_baffles = sys.calc_baffles(
                        baffles_blank=self.baffles_blank,
                        transform_extra=sys.transform,
                    )
                    baffles = BaffleList([b1.unary_union(b2) for b1, b2 in zip(baffles, new_baffles)])
            self._baffles_cache = baffles
        return self._baffles_cache

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[str, str] = ('x', 'y'),
            transform_extra: typ.Optional[transforms.TransformList] = None,
            plot_rays: bool = True,
            color_axis: str = 'wavelength',
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
