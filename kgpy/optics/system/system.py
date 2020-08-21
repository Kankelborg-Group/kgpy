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
from .. import ZemaxCompatible, Rays, material, surface, aperture, coordinate

__all__ = ['System']

SurfacesT = typ.TypeVar('SurfacesT', bound=typ.Union[typ.Iterable[surface.Surface], ZemaxCompatible])


def default_field_mask_func(fx: u.Quantity, fy: u.Quantity) -> np.ndarray:
    fx, fy = np.broadcast_arrays(fx, fy, subok=True)
    return np.ones(fx.shape, dtype=np.bool)


@dataclasses.dataclass
class System(ZemaxCompatible, kgpy.mixin.Broadcastable, kgpy.mixin.Named, typ.Generic[SurfacesT]):
    object_surface: surface.ObjectSurface = dataclasses.field(default_factory=lambda: surface.ObjectSurface())
    surfaces: SurfacesT = dataclasses.field(default_factory=lambda: [])
    stop_surface: typ.Optional[surface.Standard] = None
    wavelengths: typ.Optional[u.Quantity] = None
    pupil_samples: typ.Union[int, typ.Tuple[int, int]] = 3
    pupil_margin: u.Quantity = 1 * u.um
    field_min: typ.Optional[u.Quantity] = None
    field_max: typ.Optional[u.Quantity] = None
    field_samples: typ.Union[int, typ.Tuple[int, int]] = 3
    field_mask_func: typ.Callable[[u.Quantity, u.Quantity], np.ndarray] = default_field_mask_func
    baffle_positions: typ.Optional[typ.List[coordinate.TiltTranslate]] = None

    def __post_init__(self):
        self.update()

    def to_zemax(self) -> 'System':
        from kgpy.optics import zemax
        return zemax.System(
            name=self.name,
            surfaces=self.surfaces.to_zemax(),
        )

    def update(self) -> typ.NoReturn:
        self._rays_input_cache = None
        self._raytrace_cache = None

    @property
    def standard_surfaces(self) -> typ.Iterator[surface.Standard]:
        for s in self.surfaces:
            if isinstance(s, surface.Standard):
                if s.is_active:
                    yield s

    @property
    def aperture_surfaces(self) -> typ.Iterator[surface.Standard]:
        for s in self.standard_surfaces:
            if s.aperture is not None:
                if s.aperture.is_active:
                    yield s

    @property
    def test_stop_surfaces(self) -> typ.Iterator[surface.Standard]:
        for s in self.aperture_surfaces:
            if s.aperture.is_test_stop:
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
        return list(self)[~0]

    @property
    def config_broadcast(self):
        all_surface_battrs = None
        for s in self.surfaces:
            all_surface_battrs = np.broadcast(all_surface_battrs, s.config_broadcast)
            all_surface_battrs = np.broadcast_to(np.array(1), all_surface_battrs.shape)

        return all_surface_battrs

    def __iter__(self) -> typ.Iterator[surface.Surface]:
        old_surf = self.object_surface
        yield from old_surf
        for surf in self.surfaces:
            surf.previous_surface = old_surf
            yield surf
            old_surf = surf

    @property
    def field_x(self) -> u.Quantity:
        return kgpy.linspace(self.field_min[x], self.field_max[x], self.field_samples_normalized[ix], axis=~0)

    @property
    def field_y(self) -> u.Quantity:
        return kgpy.linspace(self.field_min[y], self.field_max[y], self.field_samples_normalized[iy], axis=~0)

    def pupil_x(self, surf: surface.Standard) -> u.Quantity:
        aper = surf.aperture
        return kgpy.linspace(
            start=aper.min[x] + self.pupil_margin,
            stop=aper.max[x] - self.pupil_margin,
            num=self.pupil_samples_normalized[ix],
            axis=~0,
        )

    def pupil_y(self, surf: surface.Standard) -> u.Quantity:
        aper = surf.aperture
        return kgpy.linspace(
            start=aper.min[y] + self.pupil_margin,
            stop=aper.max[y] - self.pupil_margin,
            num=self.pupil_samples_normalized[iy],
            axis=~0,
        )

    def raytrace(self, rays: Rays) -> typ.List[surface.Surface]:
        # surfaces_orig = list(self)
        # surfaces = []
        # for s in surfaces_orig:
        #     surfaces.append(s.copy())
        surfaces = list(self)
        surfaces[0].rays_input = rays
        surfaces[0].update()
        return surfaces

    @property
    def surfaces_raytraced(self) -> typ.List[surface.Surface]:
        if self._raytrace_cache is None:
            self._raytrace_cache = self._surfaces_raytraced
        return self._raytrace_cache

    @property
    def _surfaces_raytraced(self) -> typ.List[surface.Surface]:
        return self.raytrace(self.rays_input)

    @property
    def rays_output(self):
        return self.surfaces_raytraced[~0].rays_output

    @property
    def rays_input(self):
        if self._rays_input_cache is None:
            self._rays_input_cache = self._rays_input
        return self._rays_input_cache

    @property
    def _rays_input(self) -> Rays:

        if np.isinf(self.object_surface.thickness).all():

            position_guess = kgpy.vector.from_components(use_z=False) << u.mm

            step_size = 1 * u.nm
            step = kgpy.vector.from_components(ax=step_size, ay=step_size, use_z=False)

            for surf in self.test_stop_surfaces:
                surf_index = list(self).index(surf)
                px, py = self.pupil_x(surf), self.pupil_y(surf)
                target_position = kgpy.vector.from_components(px[..., None], py)

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
                    rays = self.raytrace(rays)[surf_index].rays_output
                    return (rays.position - target_position)[xy]

                position_guess = kgpy.optimization.root_finding.secant(
                    func=position_error,
                    root_guess=position_guess,
                    step_size=step,
                    max_abs_error=1 * u.nm,
                    max_iterations=100,
                )

                if surf == self.stop_surface:
                    break

            input_rays = Rays.from_field_angles(
                wavelength_grid=self.wavelengths,
                position=kgpy.vector.to_3d(position_guess),
                field_grid_x=self.field_x,
                field_grid_y=self.field_y,
                field_mask_func=self.field_mask_func,
                pupil_grid_x=self.pupil_x,
                pupil_grid_y=self.pupil_y,
            )

        else:

            direction_guess = kgpy.vector.from_components(use_z=False) << u.deg

            step_size = 1e-10 * u.deg

            step = kgpy.vector.from_components(ax=step_size, ay=step_size, use_z=False)

            for surf in self.test_stop_surfaces:
                surf_index = list(self).index(surf)
                px, py = self.pupil_x(surf), self.pupil_y(surf)
                target_position = kgpy.vector.from_components(px[..., None], py)

                def position_error(direc: u.Quantity) -> u.Quantity:
                    direction = np.zeros(self.field_samples_normalized + target_position.shape)
                    direction[z] = 1
                    direction = kgpy.vector.rotate_x(direction, direc[y])
                    direction = kgpy.vector.rotate_y(direction, direc[x])
                    rays = Rays.from_field_positions(
                        wavelength_grid=self.wavelengths,
                        direction=direction,
                        field_grid_x=self.field_x,
                        field_grid_y=self.field_y,
                        field_mask_func=self.field_mask_func,
                        pupil_grid_x=px,
                        pupil_grid_y=py,
                    )
                    surfaces_rays = self.raytrace(rays)
                    # fig, axs = plt.subplots(nrows=2, sharex='all')
                    # self.plot_surfaces(ax=axs[0], surfaces=surfaces_rays, components=(iz, iy), plot_vignetted=True)
                    # self.plot_surfaces(ax=axs[1], surfaces=surfaces_rays, components=(iz, ix), plot_vignetted=True)
                    # plt.show()
                    rays = surfaces_rays[surf_index].rays_output
                    return (rays.position - target_position)[xy]

                direction_guess = kgpy.optimization.root_finding.secant(
                    func=position_error,
                    root_guess=direction_guess,
                    step_size=step,
                    max_abs_error=1 * u.nm,
                    max_iterations=100,
                )

                if surf == self.stop_surface:
                    break

            direction = kgpy.vector.from_components(az=1)
            direction = kgpy.vector.rotate_x(direction, direction_guess[y])
            direction = kgpy.vector.rotate_y(direction, direction_guess[x])
            input_rays = Rays.from_field_positions(
                wavelength_grid=self.wavelengths,
                direction=direction,
                field_grid_x=self.field_x,
                field_grid_y=self.field_y,
                field_mask_func=self.field_mask_func,
                pupil_grid_x=self.pupil_x(self.stop_surface),
                pupil_grid_y=self.pupil_y(self.stop_surface),
            )

        return input_rays

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

    def _calc_baffles(self, baffle_positions):

        pass

    def print_surfaces(self) -> typ.NoReturn:
        for surf in self:
            print(surf)

    def plot_footprint(
            self,
            ax: typ.Optional[plt.Axes] = None,
            surf: typ.Optional[surface.Standard] = None,
            color_axis: int = Rays.axis.wavelength,
            plot_apertures: bool = True,
            plot_vignetted: bool = False,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()

        if surf is None:
            surf = self.image_surface

        surf_index = list(self).index(surf)
        rays = self.surfaces_raytraced[surf_index].rays_output.copy()
        rays.vignetted_mask = self.rays_output.vignetted_mask

        rays.plot_position(ax=ax, color_axis=color_axis, plot_vignetted=plot_vignetted)

        if plot_apertures:
            surf.plot_2d(ax)

        return ax

    def plot_projections(
            self,
            start_surface: typ.Optional[surface.Surface] = None,
            final_surface: typ.Optional[surface.Surface] = None,
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
            (kgpy.vector.ix, kgpy.vector.iy),
            (kgpy.vector.iz, kgpy.vector.iy),
            (kgpy.vector.iz, kgpy.vector.ix),
        ]
        for ax_index, plane in zip(ax_indices, planes):
            self.plot_2d(
                ax=axs[ax_index],
                components=plane,
                start_surface=start_surface,
                final_surface=final_surface,
                color_axis=color_axis,
                plot_vignetted=plot_vignetted,
                plot_rays=plot_rays,
            )
            axs[ax_index].get_legend().remove()

        handles, labels = axs[xy].get_legend_handles_labels()
        label_dict = dict(zip(labels, handles))
        fig.legend(label_dict.values(), label_dict.keys())

        return fig

    def plot_2d(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[int, int] = (kgpy.vector.ix, kgpy.vector.iy),
            start_surface: typ.Optional[surface.Surface] = None,
            final_surface: typ.Optional[surface.Surface] = None,
            color_axis: int = Rays.axis.wavelength,
            plot_vignetted: bool = False,
            plot_rays: bool = True,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()

        surfaces = self.surfaces_raytraced

        if start_surface is None:
            start_surface = surfaces[0]

        if final_surface is None:
            final_surface = surfaces[~0]

        start_surface_index = surfaces.index(start_surface)
        end_surface_index = surfaces.index(final_surface)

        return self.plot_surfaces(
            ax=ax,
            components=components,
            surfaces=surfaces[start_surface_index:end_surface_index + 1],
            color_axis=color_axis,
            plot_vignetted=plot_vignetted,
            plot_rays=plot_rays,
        )

    @staticmethod
    def plot_surfaces(
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[int, int] = (kgpy.vector.ix, kgpy.vector.iy),
            surfaces: typ.Optional[typ.List[surface.Surface]] = None,
            color_axis: int = Rays.axis.wavelength,
            plot_vignetted: bool = False,
            plot_rays: bool = True,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()

        for surf in surfaces:
            surf.plot_2d(ax=ax, components=components)

        if plot_rays:
            intercepts = []
            for surf in surfaces:
                intercept = surf.transform_to_global(surf.rays_output.position, num_extra_dims=5)
                intercepts.append(intercept)
            intercepts = u.Quantity(intercepts)

            img_rays = surfaces[~0].rays_output

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

            ax.set_xlim(right=1.1 * ax.get_xlim()[1])
            handles, labels = ax.get_legend_handles_labels()
            label_dict = dict(zip(labels, handles))
            ax.legend(label_dict.values(), label_dict.keys(), loc='upper right')

        return ax

    def to_occ(self):

        occ_unit = u.mm

        import OCC.Core.TopoDS
        import OCC.Core as occ
        from OCC.Core import gp, Geom, BRepBuilderAPI, BRepPrimAPI, Standard, BRepLib, Geom2d, BRep, Poly, TColgp, TopLoc


        occ_shapes = []

        for surf in self.aperture_surfaces:

            print(surf)


            # wire = surf.transform_to_global(surf.aperture.wire, self, num_extra_dims=1)
            # wire = surf.aperture.global_wire(self, surf,  apply_sag=True)
            wire = surf.aperture.wire
            wire = np.broadcast_to(wire, self.shape + wire.shape, subok=True)
            if surf.is_sphere:
                wire = wire / surf.radius
            for c, wire_c in enumerate(wire):
                occ_arr = TColgp.TColgp_Array1OfPnt2d(0, len(wire_c))
                # occ_wire = BRepBuilderAPI.BRepBuilderAPI_MakeWire()
                # occ_poly = OCC.Core.BRepBuilderAPI.BRepBuilderAPI_MakePolygon()
                # old_occ_point = OCC.Core.gp.gp_Pnt2d(*wire_c[0][xy].value)
                for p, point in enumerate(wire_c):
                    occ_point = OCC.Core.gp.gp_Pnt2d(*point[xy].value)
                    occ_arr.SetValue(p, occ_point)
                    # occ_wire.Add(BRepBuilderAPI.BRepBuilderAPI_MakeEdge2d(old_occ_point, occ_point).Edge())
                    old_occ_point = occ_point
                occ_poly2d = Poly.Poly_Polygon2D(occ_arr)
                # occ_wire = occ_wire.Wire()

                # occ_wire = occ_poly.Wire()
                # occ_shapes.append(occ_wire)



                if surf.is_plane:

                    length = 1 * occ_unit
                    point_0 = surf.transform_to_global(kgpy.vector.from_components() << occ_unit, self)[c]
                    point_1 = surf.transform_to_global(kgpy.vector.from_components(az=length), self)[c]
                    normal = (point_1 - point_0) / length
                    occ_point_0 = gp.gp_Pnt(*point_0.value)
                    occ_normal = gp.gp_Dir(*normal.value)
                    occ_ax3 = gp.gp_Ax3(occ_point_0, occ_normal)
                    occ_surf = Geom.Geom_Plane(occ_ax3)
                    occ_polysurf = BRep.BRep_PolygonOnSurface(occ_poly2d, occ_surf, TopLoc.TopLoc_Location())
                    # occ_face = BRepBuilderAPI.BRepBuilderAPI_MakeFace(occ_surf, occ_wire).Face()
                    occ_shapes.append(occ_polysurf.Curve3D())

                elif surf.is_sphere:

                    point_0 = surf.transform_to_global(kgpy.vector.from_components() << occ_unit, self)[c]
                    point_1 = surf.transform_to_global(kgpy.vector.from_components(ax=surf.radius), self)[c]
                    point_2 = surf.transform_to_global(kgpy.vector.from_components(ay=surf.radius), self)[c]
                    point_3 = surf.transform_to_global(kgpy.vector.from_components(az=surf.radius), self)[c]
                    xhat = (point_1 - point_0) / surf.radius
                    yhat = (point_2 - point_0) / surf.radius
                    zhat = (point_3 - point_0) / surf.radius
                    occ_point_3 = gp.gp_Pnt(*point_3.to(occ_unit).value)
                    occ_xhat = gp.gp_Dir(*xhat.value)
                    occ_yhat = gp.gp_Dir(*yhat.value)
                    occ_zhat = gp.gp_Dir(*zhat.value)
                    # occ_ax2 = gp.gp_Ax2(occ_point_3, occ_xhat, occ_zhat)
                    occ_ax3 = gp.gp_Ax3(occ_point_3, occ_zhat)
                    # occ_curve = Geom.Geom_Circle(occ_ax2, surf.radius.to(occ_unit).value)
                    # occ_curve = Geom.Geom_TrimmedCurve(occ_curve, 3 * np.pi / 4, np.pi,)
                    # rev_ax = gp.gp_Ax1(occ_point_3, occ_zhat)
                    # occ_surf = Geom.Geom_SurfaceOfRevolution(occ_curve, rev_ax)
                    # occ_surf = gp.gp_Sphere(occ_ax3, surf.radius.to(occ_unit).value)
                    occ_surf = Geom.Geom_SphericalSurface(occ_ax3, surf.radius.to(occ_unit).value)
                    occ_polysurf = BRep.BRep_PolygonOnSurface(occ_poly2d, occ_surf, TopLoc.TopLoc_Location())
                    # occ_surf = Geom.Geom_CylindricalSurface(occ_ax3, surf.radius.to(occ_unit).value)
                    # occ_surf = BRepPrimAPI.BRepPrimAPI_MakeSphere(surf.radius.to(occ_unit).value)
                    # occ_surf = BRepBuilderAPI.BRepBuilderAPI_MakeFace(occ_surf).Face()
                    # occ_face = BRepBuilderAPI.BRepBuilderAPI_MakeFace(occ_surf, occ_polysurf).Face()
                    # BRepLib.breplib_BuildCurves3d(occ_face)

                    # occ_solid = BRepPrimAPI.BRepPrimAPI_MakePrism(occ_face, gp.gp_Vec(*(10 * zhat).value)).Shape()
                    # occ_surf = Geom.Geom_RectangularTrimmedSurface(occ_surf, -np.pi / 2, -np.pi/4, False)
                    occ_shapes.append(occ_polysurf.Surface2())

        return occ_shapes

    def plot_occ(self):

        import OCC.Display.SimpleGui

        display, start_display, add_menu, add_function_to_menu = OCC.Display.SimpleGui.init_display()

        occ_shapes = self.to_occ()

        for shape in occ_shapes:
            display.DisplayShape(shape)

        start_display()








