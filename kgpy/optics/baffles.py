import typing as typ
import abc
import collections
import dataclasses
import pathlib
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import astropy.units as u
import shapely.geometry
import shapely.ops
import ezdxf.addons
import kgpy.mixin
import kgpy.vectors
import kgpy.transforms
import kgpy.geometry
from . import rays
from . import surfaces

__all__ = ['Baffle', 'BaffleList']

ObscurationT = typ.TypeVar('ObscurationT', bound=surfaces.apertures.Polygon)


@dataclasses.dataclass
class Baffle(
    kgpy.mixin.Broadcastable,
    kgpy.mixin.Plottable,
    kgpy.transforms.Transformable,
    kgpy.mixin.Named,
    typ.Generic[ObscurationT],
):

    apertures_base: typ.List[surfaces.apertures.IrregularPolygon] = dataclasses.field(default_factory=lambda: [])
    apertures_extra: typ.List[surfaces.apertures.Aperture] = dataclasses.field(default_factory=lambda: [])
    obscuration_base: typ.Optional[ObscurationT] = None
    margin: u.Quantity = 1 * u.mm
    min_distance: u.Quantity = 2 * u.mm
    combined_axes: typ.Optional[typ.Sequence[int]] = None
    shapely_unit: u.Unit = u.mm
    buffer_cap_style: int = shapely.geometry.CAP_STYLE.square
    buffer_join_style: int = shapely.geometry.JOIN_STYLE.mitre
    buffer_resolution = 1

    def concat_apertures_from_raytrace(
            self,
            raytrace: rays.RayFunctionList,
            transform_extra: typ.Optional[kgpy.transforms.TransformList] = None,
            hull_axes: typ.Optional[typ.Sequence[int]] = None,
            color: str = 'black',
    ) -> 'Baffle':

        if transform_extra is None:
            transform_extra = kgpy.transforms.TransformList()

        img_rays = raytrace[~0]

        position = raytrace.intercepts
        position = transform_extra(position, num_extra_dims=img_rays.axis.ndim)

        position = position.reshape(position.shape[:1] + img_rays.shape + (-1, ))
        mask = img_rays.mask.reshape(img_rays.shape + (-1, 1))
        position_1, position_2 = position[:~0], position[1:]
        position_1, position_2 = np.moveaxis(position_1, 0, ~0), np.moveaxis(position_2, 0, ~0)

        return self.concat_apertures_from_global_positions(
            position_1=position_1,
            position_2=position_2,
            mask=mask,
            hull_axes=hull_axes,
            color=color,
        )

    def concat_apertures_from_lofts(
            self,
            lofts: typ.Dict[int, typ.Tuple[surfaces.Surface, surfaces.Surface]],
            transform_extra: typ.Optional[kgpy.transforms.TransformList] = None,
            color: str = 'black',
    ) -> 'Baffle':

        if transform_extra is None:
            transform_extra = kgpy.transforms.TransformList()

        result = self

        for loft in lofts.values():
            surf_0, surf_1 = loft
            transform_0 = transform_extra + surf_0.transform
            transform_1 = transform_extra + surf_1.transform
            vertices_0, vertices_1 = surf_0.aperture.vertices, surf_1.aperture.vertices
            if surf_0.sag is not None:
                vertices_0.z = vertices_0.z + surf_0.sag(vertices_0.x, vertices_0.y)
            if surf_1.sag is not None:
                vertices_1.z = vertices_1.z + surf_1.sag(vertices_1.x, vertices_1.y)
            position_0 = transform_0(vertices_0, num_extra_dims=1)[..., :, np.newaxis, np.newaxis]
            position_1 = transform_1(vertices_1, num_extra_dims=1)[..., np.newaxis, :, np.newaxis]
            result = result.concat_apertures_from_global_positions(
                position_1=position_0, position_2=position_1, color=color)

        return result

    def concat_apertures_from_global_positions(
            self,
            position_1: kgpy.vectors.Cartesian3D,
            position_2: kgpy.vectors.Cartesian3D,
            mask: typ.Optional = None,
            hull_axes: typ.Optional[typ.Sequence[int]] = None,
            color: str = 'black',
    ) -> 'Baffle':

        position_1 = self.transform.inverse(position_1)
        position_2 = self.transform.inverse(position_2)

        intercept = kgpy.geometry.segment_plane_intercept(
            plane_point=kgpy.vectors.Cartesian3D() * u.mm,
            plane_normal=kgpy.vectors.Cartesian3D.z_hat(),
            line_point_1=position_1,
            line_point_2=position_2,
        )

        return self.concat_apertures_from_intercept(intercept=intercept, mask=mask, hull_axes=hull_axes, color=color)

    def concat_apertures_from_intercept(
            self,
            intercept: kgpy.vectors.Cartesian3D,
            mask: u.Quantity,
            hull_axes: typ.Optional[typ.Sequence[int]] = None,
            color: str = 'black',
    ) -> 'Baffle':
        sh = intercept.shape

        if mask is None:
            mask = True
        mask = mask & np.isfinite(intercept.length_l1)

        num_axes = len(sh[:~0])

        if hull_axes is None:
            hull_axes = tuple(range(num_axes))
        else:
            hull_axes = [ax % num_axes for ax in hull_axes]

        num_hull_axes = len(hull_axes)
        hull_axes_dest = list(range(num_hull_axes))
        intercept = np.moveaxis(intercept, hull_axes, hull_axes_dest)
        mask = np.moveaxis(mask, hull_axes, hull_axes_dest)

        intercept = intercept.reshape((-1, ) + intercept.shape[num_hull_axes:])
        mask = mask.reshape((-1, ) + mask.shape[num_hull_axes:])

        intercept = np.moveaxis(intercept, 0, ~0)
        mask = np.moveaxis(mask, 0, ~0)

        intercept = intercept.reshape((-1, ) + intercept.shape[~0:])
        mask = mask.reshape((-1, ) + mask.shape[~0:])

        apertures = []
        for i in range(intercept.shape[0]):
            points = intercept[i, mask[i]]
            if points.shape[0] > 2:
                # points = shapely.geometry.MultiPoint(points.quantity.to(self.shapely_unit).value)
                # poly = points.convex_hull
                # aper = self._to_aperture(poly)
                hull = scipy.spatial.ConvexHull(points.xy.quantity)
                aper = surfaces.apertures.IrregularPolygon(vertices=points[hull.vertices].copy())
                aper.color = color
                apertures.append(aper)

        return self.concat_apertures(apertures)

    def concat_apertures(self, apertures: typ.List[surfaces.apertures.IrregularPolygon]) -> 'Baffle':
        # apertures = np.broadcast_to(apertures, self.shape[:~0] + apertures.shape[~0:])
        other = self.copy()
        other.apertures_base += apertures
        return other

    @property
    def interiors(self) -> shapely.geometry.MultiPolygon:

        apertures = self._to_shapely_multipoly(self.apertures_base)

        margin = self.margin.to(self.shapely_unit).value
        apertures = [aper.buffer(margin, **self._buffer_kwargs) for aper in apertures]

        # dist = self.min_distance.to(self.shapely_unit).value / 2
        # apertures = [aper.buffer(dist, **self._buffer_kwargs) for aper in apertures]
        # apertures = shapely.ops.unary_union(apertures)
        # if isinstance(apertures, shapely.geometry.Polygon):
        #     apertures = shapely.geometry.MultiPolygon([apertures])
        #
        # apertures = [aper.buffer(-dist, **self._buffer_kwargs) for aper in apertures]
        return shapely.geometry.MultiPolygon(apertures)

    @property
    def apertures(self):
        apertures = []
        for interior in self._shapely_baffle.interiors:
            # a = optics.surface.aperture.IrregularPolygon(vertices=vector.to_3d(np.array(interior) << self.shapely_unit))
            a = surfaces.apertures.IrregularPolygon(
                vertices=vector.Vector2D.from_quantity(interior << self.shapely_unit).to_3d()
            )
            apertures.append(a)
        return apertures

    @property
    def obscuration(self):
        return self._to_aperture(self._shapely_baffle)

    def _to_shapely_poly(self, aperture: surfaces.apertures.Polygon) -> shapely.geometry.Polygon:
        return shapely.geometry.Polygon(shell=aperture.vertices.to(self.shapely_unit).quantity.value)

    def _to_shapely_multipoly(self, aperture_list: typ.List[surfaces.apertures.Polygon]) -> shapely.geometry.MultiPolygon:
        return shapely.geometry.MultiPolygon([self._to_shapely_poly(aper) for aper in aperture_list])

    def _to_aperture(self, aperture: shapely.geometry.Polygon) -> surfaces.apertures.IrregularPolygon:
        return surfaces.apertures.IrregularPolygon(
            # vertices=vector.to_3d(np.array(aperture.exterior) << self.shapely_unit))
            vertices=vector.Vector2D.from_quantity(aperture.exterior << self.shapely_unit).to_3d()
        )

    def _to_aperture_list(self, apertures: shapely.geometry.MultiPolygon) -> typ.List[surfaces.apertures.Polygon]:
        return [self._to_aperture(aper) for aper in apertures]

    @property
    def _shapely_baffle(self) -> shapely.geometry.Polygon:
        dist = self.min_distance.to(self.shapely_unit).value / 2
        result = shapely.geometry.Polygon(
            shell=self._to_shapely_poly(self.obscuration_base).exterior,
            holes=[aper.exterior for aper in self.interiors]
        )
        cap_style = shapely.geometry.CAP_STYLE.square
        join_style = shapely.geometry.JOIN_STYLE.mitre
        result = result.buffer(-dist, cap_style=cap_style, join_style=join_style)
        result = result.buffer(dist, cap_style=cap_style, join_style=join_style)
        return result

    @property
    def _buffer_kwargs(self) -> typ.Dict[str, typ.Any]:
        return {
            'resolution': self.buffer_resolution,
            'cap_style': self.buffer_cap_style,
            'join_style': self.buffer_join_style,
        }

    def unary_union(self, other: 'Baffle'):

        if self.obscuration_base != other.obscuration_base:
            raise ValueError('Must have the same base obscuration to evaluate union')

        if self.transform != other.transform:
            raise ValueError('Must have same location to evaluate union')

        if self.margin != other.margin:
            raise ValueError

        if self.min_distance != other.min_distance:
            raise ValueError

        return self.concat_apertures(other.apertures_base)

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            component_x: str = 'x',
            component_y: str = 'y',
            component_z: str = 'z',
            transform_extra: typ.Optional[kgpy.transforms.TransformList] = None,
            to_global: bool = False,
            plot_apertures_base: bool = False,
            **kwargs,
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()

        kwargs = {**self.plot_kwargs, **kwargs}

        if to_global:
            if transform_extra is None:
                transform_extra = kgpy.transforms.TransformList()
            transform_extra = transform_extra + self.transform

        if self.apertures is not None:
            for aper in self.apertures:
                aper.plot(
                    ax=ax,
                    component_x=component_x,
                    component_y=component_y,
                    component_z=component_z,
                    transform_extra=transform_extra,
                    **kwargs,
                )

        for aper in self.apertures_extra:
            aper.plot(
                ax=ax,
                component_x=component_x,
                component_y=component_y,
                component_z=component_z,
                transform_extra=transform_extra,
                **kwargs,
            )

        if self.obscuration is not None:
            self.obscuration.plot(
                ax=ax,
                component_x=component_x,
                component_y=component_y,
                component_z=component_z,
                transform_extra=transform_extra,
                **kwargs,
            )

        if plot_apertures_base:
            for aper in self.apertures_base:
                aper.plot(
                    ax=ax,
                    component_x=component_x,
                    component_y=component_y,
                    component_z=component_z,
                    transform_extra=transform_extra,
                    **kwargs,
                )

        return ax

    def to_dxf(self, filename: pathlib.Path, dxf_unit: u.Unit = u.imperial.inch):
        if self.obscuration is not None:
            with ezdxf.addons.r12writer(filename) as dxf:

                if self.obscuration is not None:
                    dxf.add_polyline(self.obscuration.vertices.quantity.to(dxf_unit).value, closed=True)

                if self.apertures is not None:
                    for aper in self.apertures:
                        if isinstance(aper, surfaces.apertures.Polygon):
                            dxf.add_polyline(aper.vertices.quantity.to(dxf_unit).value, closed=True)
                        elif isinstance(aper, surfaces.apertures.Circular):
                            dxf.add_circle(
                                (aper.descenter.x.to(dxf_unit).value, aper.decenter.y.to(dxf_unit).value),
                                aper.radius.to(dxf_unit).value
                            )
                        else:
                            raise NotImplementedError

                if self.apertures_extra is not None:
                    for aper in self.apertures_extra:
                        if isinstance(aper, surfaces.apertures.Polygon):
                            dxf.add_polyline(aper.vertices.quantity.to(dxf_unit).value, closed=True)
                        elif isinstance(aper, surfaces.apertures.Circular):
                            dxf.add_circle(
                                (aper.decenter.x.to(dxf_unit).value, aper.decenter.y.to(dxf_unit).value),
                                aper.radius.to(dxf_unit).value
                            )
                        else:
                            raise NotImplementedError


class BaffleList(
    collections.UserList,
):

    def concat_apertures_from_raytrace(
            self,
            raytrace: rays.RayFunctionList,
            transform_extra: typ.Optional[kgpy.transforms.TransformList] = None,
            hull_axes: typ.Optional[typ.Sequence[int]] = None,
            color: str = 'black',
    ) -> 'BaffleList':
        baffle_list = []
        for b in self:
            baffle_list.append(b.concat_apertures_from_raytrace(
                raytrace=raytrace,
                transform_extra=transform_extra,
                hull_axes=hull_axes,
                color=color,
            ))
        return BaffleList(baffle_list)

    def concat_apertures_from_lofts(
            self,
            lofts: typ.Dict[int, typ.Tuple[surfaces.Surface, surfaces.Surface]],
            transform_extra: typ.Optional[kgpy.transforms.TransformList] = None,
            color: str = 'black',
    ) -> 'BaffleList':
        baffle_list = []
        for b in self:
            baffle_list.append(b.concat_apertures_from_lofts(
                lofts=lofts,
                transform_extra=transform_extra,
                color=color,
            ))
        return BaffleList(baffle_list)

    def concat_apertures_from_global_positions(
            self,
            position_1: kgpy.vectors.Cartesian3D,
            position_2: kgpy.vectors.Cartesian3D,
            mask: typ.Optional = None,
            hull_axes: typ.Optional[typ.Sequence[int]] = None,
    ) -> 'BaffleList':
        baffle_list = []
        for b in self:
            baffle_list.append(b.concat_apertures_from_global_positions(
                position_1=position_1,
                position_2=position_2,
                mask=mask,
                hull_axes=hull_axes,
            ))
        return BaffleList(baffle_list)

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[str, str] = ('x', 'y'),
            plot_kwargs: typ.Optional[typ.Dict[str, typ.Any]] = None,
            transform_extra: typ.Optional[kgpy.transforms.TransformList] = None,
            plot_apertures_base: bool = False
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()

        if transform_extra is None:
            transform_extra = kgpy.transforms.TransformList()

        for baffle in self:
            baffle.plot(
                ax=ax,
                components=components,
                plot_kwargs=plot_kwargs,
                transform_extra=transform_extra,
                to_global=True,
                plot_apertures_base=plot_apertures_base
            )

        return ax

    def to_dxf(self, file_base: pathlib.Path):
        for i in range(len(self)):
            filename = file_base.parent / (str(file_base.name) + '_' + str(i) + '.dxf')
            self[i].to_dxf(filename)
