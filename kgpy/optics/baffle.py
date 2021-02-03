import typing as typ
import abc
import collections
import dataclasses
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import shapely.geometry
import shapely.ops
import ezdxf.addons
from kgpy import mixin, vector, transform as tfrm, geometry, optics
from . import rays, surface

__all__ = ['Baffle', 'BaffleList']

ObscurationT = typ.TypeVar('ObscurationT', bound=surface.aperture.Polygon)


@dataclasses.dataclass
class Baffle(
    mixin.Broadcastable,
    mixin.Colorable,
    tfrm.rigid.Transformable,
    mixin.Named,
    typ.Generic[ObscurationT],
):

    apertures_base: typ.List[surface.aperture.IrregularPolygon] = dataclasses.field(default_factory=lambda: [])
    apertures_extra: typ.List[surface.aperture.Aperture] = dataclasses.field(default_factory=lambda: [])
    obscuration_base: typ.Optional[ObscurationT] = None
    margin: u.Quantity = 1 * u.mm
    min_distance: u.Quantity = 2 * u.mm
    combined_axes: typ.Optional[typ.Sequence[int]] = None
    shapely_unit: u.Unit = u.mm
    buffer_cap_style: int = shapely.geometry.CAP_STYLE.square
    buffer_join_style: int = shapely.geometry.JOIN_STYLE.mitre
    buffer_resolution = 10

    # def concat_apertures_from_raytrace(
    #         self,
    #         raytrace: rays.RaysList,
    # ) -> 'Baffle':
    #
    #     img_rays = raytrace[~0]
    #
    #     position_shape = (len(raytrace), ) + img_rays.vector_grid_shape
    #     position = np.empty(position_shape) << img_rays.position.unit
    #
    #     for i in range(position_shape[0]):
    #         r = raytrace[i]
    #         position[i] = r.transform.inverse(r.position, num_extra_dims=5)
    #
    #     position = position.reshape(position_shape[:1] + img_rays.shape + (-1, 3))
    #     mask = img_rays.mask.reshape(img_rays.shape + (1, -1, ))
    #     position_1, position_2 = position[:~0], position[1:]
    #     position_1, position_2 = np.moveaxis(position_1, 0, ~2), np.moveaxis(position_2, 0, ~2)
    #
    #     return self.concat_apertures_from_global_positions(position_1, position_2, mask=mask)

    def concat_apertures_from_global_positions(
            self,
            position_1: vector.Vector3D,
            position_2: vector.Vector3D,
            mask: typ.Optional = None,
            hull_axes: typ.Optional[typ.Sequence[int]] = None,
    ) -> 'Baffle':

        position_1 = self.transform.inverse(position_1, num_extra_dims=1)
        position_2 = self.transform.inverse(position_2, num_extra_dims=1)

        intercept = geometry.segment_plane_intercept(
            plane_point=vector.Vector3D() * u.mm,
            plane_normal=vector.z_hat,
            line_point_1=position_1,
            line_point_2=position_2,
        )

        return self.concat_apertures_from_intercept(intercept=intercept, mask=mask, hull_axes=hull_axes)

    def concat_apertures_from_intercept(
            self,
            intercept: vector.Vector3D,
            mask: u.Quantity,
            hull_axes: typ.Optional[typ.Sequence[int]] = None,
    ) -> 'Baffle':
        sh = intercept.shape

        if mask is None:
            mask = True
        mask = mask & np.isfinite(intercept.length_l1)

        num_axes = len(sh[:~0])
        num_hull_axes = len(hull_axes)

        if hull_axes is None:
            hull_axes = tuple(range(num_axes))
        else:
            hull_axes = [ax % num_axes for ax in hull_axes]

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
        for i in range(len(intercept)):
            points = intercept[i, mask[i]]
            if len(points) > 2:
                points = shapely.geometry.MultiPoint(points.to(self.shapely_unit).value)
                poly = points.convex_hull
                apertures.append(self._to_aperture(poly))

        return self.concat_apertures(apertures)

    def concat_apertures(self, apertures: typ.List[surface.aperture.IrregularPolygon]) -> 'Baffle':
        # apertures = np.broadcast_to(apertures, self.shape[:~0] + apertures.shape[~0:])
        other = self.copy()
        other.apertures_base += apertures
        return other

    @property
    def interiors(self) -> shapely.geometry.MultiPolygon:

        # combined_axes = self.combined_axes
        # if combined_axes is None:
        #     combined_axes = tuple(range(len(self.shape)))
        #
        # apertures = self.apertures_base
        # sh = apertures.shape
        # new_sh = list(sh)
        # for axis in combined_axes:
        #     apertures = np.moveaxis(apertures, axis, ~0)
        #     new_sh[axis] = 1
        #
        # num_combined_axes = len(combined_axes)
        # apertures = apertures.reshape(apertures.shape[:~(num_combined_axes - 1)] + (-1, ))
        #
        # apertures = apertures.reshape((-1, ) + apertures.shape[~0:])

        # apertures = [self._to_shapely_multipoly(aper).convex_hull for aper in self.apertures_base]
        apertures = self._to_shapely_multipoly(self.apertures_base)

        margin = self.margin.to(self.shapely_unit).value
        apertures = [aper.buffer(margin, **self._buffer_kwargs) for aper in apertures]

        dist = self.min_distance.to(self.shapely_unit).value / 2
        apertures = [aper.buffer(dist, **self._buffer_kwargs) for aper in apertures]
        apertures = shapely.ops.unary_union(apertures)
        if isinstance(apertures, shapely.geometry.Polygon):
            apertures = shapely.geometry.MultiPolygon([apertures])

        apertures = [aper.buffer(-dist, **self._buffer_kwargs) for aper in apertures]
        return shapely.geometry.MultiPolygon(apertures)

    @property
    def apertures(self):
        apertures = []
        for interior in self._shapely_baffle.interiors:
            # a = optics.surface.aperture.IrregularPolygon(vertices=vector.to_3d(np.array(interior) << self.shapely_unit))
            a = optics.surface.aperture.IrregularPolygon(
                vertices=vector.Vector2D.from_quantity(interior << self.shapely_unit).to_3d()
            )
            apertures.append(a)
        return apertures

    @property
    def obscuration(self):
        return self._to_aperture(self._shapely_baffle)

    def _to_shapely_poly(self, aperture: surface.aperture.Polygon) -> shapely.geometry.Polygon:
        return shapely.geometry.Polygon(shell=aperture.vertices.to(self.shapely_unit).value)

    def _to_shapely_multipoly(self, aperture_list: typ.List[surface.aperture.Polygon]) -> shapely.geometry.MultiPolygon:
        return shapely.geometry.MultiPolygon([self._to_shapely_poly(aper) for aper in aperture_list])

    def _to_aperture(self, aperture: shapely.geometry.Polygon) -> surface.aperture.IrregularPolygon:
        return optics.surface.aperture.IrregularPolygon(
            # vertices=vector.to_3d(np.array(aperture.exterior) << self.shapely_unit))
            vertices=vector.Vector2D.from_quantity(aperture.exterior << self.shapely_unit).to_3d()
        )

    def _to_aperture_list(self, apertures: shapely.geometry.MultiPolygon) -> typ.List[surface.aperture.Polygon]:
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

    # def _calc_convex_hulls(
    #         self,
    #         position_1: u.Quantity,
    #         position_2: u.Quantity,
    #         mask: typ.Optional[typ.List[np.ndarray]] = None
    # ) -> typ.List[shapely.geometry.Polygon]:
    #
    #     intercept = geometry.segment_plane_intercept(
    #         plane_point=[0, 0, 0] * u.mm,
    #         plane_normal=vector.z_hat,
    #         line_point_1=position_1,
    #         line_point_2=position_2,
    #     )
    #
    #     if mask is None:
    #         mask = np.ones(intercept[vector.x].shape, dtype=np.bool)
    #
    #     mask = mask & np.isfinite(intercept.sum(~0))
    #     # mask_i &= ~self.obscuration.is_unvignetted(intercept)
    #
    #     if self.union_axes is None:
    #         intercept = intercept[None, mask, :]
    #
    #     else:
    #         raise NotImplementedError
    #         # position_1 = position_1.reshape(sh + (-1, 3))
    #         # position_2 = position_2.reshape(sh + (-1, 3))
    #         #
    #         # num_union_axes = len(self.union_axes)
    #         # num_separate_axes = len(sh) - num_union_axes
    #         # union_axes_dest = num_separate_axes + np.arange(num_union_axes)
    #         #
    #         # position_1 = np.moveaxis(position_1, self.union_axes, union_axes_dest)
    #         # position_2 = np.moveaxis(position_2, self.union_axes, union_axes_dest)
    #         #
    #         # position_1 = position_1.reshape(position_1.shape[:num_separate_axes] + (-1, 3))
    #         # position_2 = position_2.reshape(position_2.shape[:num_separate_axes] + (-1, 3))
    #         #
    #         # position_1 = position_1.reshape((-1, ) + position_1.shape[~1:])
    #         # position_2 = position_2.reshape((-1, ) + position_2.shape[~1:])
    #
    #     return [shapely.geometry.MultiPoint(icpt.value).convex_hull for icpt in intercept]

    # def apertures_from_raytrace(
    #         self,
    #         raytrace: rays.RaysList,
    #         lofts: typ.List[typ.List[surface.Surface]],
    # ) -> 'Baffle':
    #
    #     shapely_unit = u.mm
    #
    #     img_rays = raytrace[~0]
    #
    #     apertures = []
    #
    #     for i in range(1, len(raytrace)):
    #         r1, r2 = raytrace[i - 1], raytrace[i]
    #         t1 = self.transform.inverse + r1.transform
    #         t2 = self.transform.inverse + r2.transform
    #         p1 = t1(r1.position, num_extra_dims=5)
    #         p2 = t2(r2.position, num_extra_dims=5)
    #         apertures += self._calc_convex_hulls(p1.to(shapely_unit), p2.to(shapely_unit), img_rays.mask)
    #
    #     for loft in lofts:
    #         for i in range(1, len(loft)):
    #             surf_1, surf_2 = loft[i - 1], loft[i]
    #             t1 = self.transform.inverse + surf_1.transform
    #             t2 = self.transform.inverse + surf_2.transform
    #             p1 = t1(surf_1.aperture.vertices, num_extra_dims=1)[..., :, None, :]
    #             p2 = t2(surf_2.aperture.vertices, num_extra_dims=1)[..., None, :, :]
    #             apertures += self._calc_convex_hulls(p1.to(shapely_unit), p2.to(shapely_unit))
    #
    #     apertures = shapely.geometry.MultiPolygon([a.buffer(self.margin.to(shapely_unit).value) for a in apertures])
    #
    #     apertures = self.combine_close_apertures(apertures, self.min_distance.to(shapely_unit).value)
    #
    #     return Baffle(
    #         name=self.name.copy(),
    #         transform=self.transform.copy(),
    #         apertures_base=self._to_aperture_list(apertures=apertures, unit=self.shapely_unit),
    #         obscuration_base=self.obscuration_base.copy(),
    #         margin=self.margin.copy(),
    #         min_distance=self.min_distance.copy(),
    #         union_axes=self.union_axes,
    #         shapely_unit=self.shapely_unit,
    #     )

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

        # apertures = self._to_shapely_multipoly(self.apertures_base + other.apertures_base, self.shapely_unit)
        #
        # apertures = self.combine_close_apertures(apertures, self.min_distance.to(self.shapely_unit).value)
        #
        # return Baffle(
        #     name=self.name.copy(),
        #     transform=self.transform.copy(),
        #     apertures_base=self._to_aperture_list(apertures=apertures, unit=self.shapely_unit),
        #     obscuration_base=self.obscuration_base.copy(),
        #     margin=self.margin.copy(),
        #     min_distance=self.min_distance.copy(),
        #     union_axes=self.union_axes,
        #     shapely_unit=self.shapely_unit,
        # )

    # @classmethod
    # def combine_close_apertures(
    #         cls,
    #         apertures: shapely.geometry.MultiPolygon,
    #         min_distance: float
    # ) -> shapely.geometry.MultiPolygon:
    #
    #     dilate_distance = min_distance / 2
    #     apertures = shapely.geometry.MultiPolygon([a.buffer(dilate_distance) for a in apertures])
    #     apertures = shapely.ops.unary_union(apertures)
    #     if isinstance(apertures, shapely.geometry.Polygon):
    #         apertures = shapely.geometry.MultiPolygon([apertures])
    #
    #     erosion_distance = -dilate_distance
    #     apertures = shapely.geometry.MultiPolygon([a.buffer(erosion_distance) for a in apertures])
    #     if isinstance(apertures, shapely.geometry.Polygon):
    #         apertures = shapely.geometry.MultiPolygon([apertures])
    #
    #     return apertures

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[str, str] = ('x', 'y'),
            color: typ.Optional[str] = None,
            transform_extra: typ.Optional[tfrm.rigid.TransformList] = None,
            to_global: bool = False,

    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()

        if color is None:
            color = self.color

        if to_global:
            if transform_extra is None:
                transform_extra = tfrm.rigid.TransformList()
            transform_extra = transform_extra + self.transform

        if self.apertures is not None:
            for aper in self.apertures:
                aper.plot(ax=ax, components=components, transform_extra=transform_extra, color=color)

        for aper in self.apertures_extra:
            aper.plot(ax=ax, components=components, transform_extra=transform_extra, color=color)

        if self.obscuration is not None:
            self.obscuration.plot(ax=ax, components=components, transform_extra=transform_extra, color=color)

        return ax

    def to_dxf(self, filename: pathlib.Path, dxf_unit: u.Unit = u.imperial.inch):
        if self.obscuration is not None:
            with ezdxf.addons.r12writer(filename) as dxf:

                if self.obscuration is not None:
                    dxf.add_polyline(self.obscuration.vertices.to(dxf_unit).value, closed=True)

                if self.apertures is not None:
                    for aper in self.apertures:
                        if isinstance(aper, optics.surface.aperture.Polygon):
                            dxf.add_polyline(aper.vertices.to(dxf_unit).value, closed=True)
                        elif isinstance(aper, optics.surface.aperture.Circular):
                            dxf.add_circle(
                                (aper.decenter.x.to(dxf_unit).value, aper.decenter.y.to(dxf_unit).value),
                                aper.radius.to(dxf_unit).value
                            )
                        else:
                            raise NotImplementedError

    def view(self) -> 'Baffle[ObscurationT]':
        other = super().view()      # type: Baffle[ObscurationT]
        other.apertures_base = self.apertures_base
        other.apertures_extra = self.apertures_extra
        other.obscuration_base = self.obscuration_base
        other.margin = self.margin
        other.min_distance = self.min_distance
        other.combined_axes = self.combined_axes
        other.shapely_unit = self.shapely_unit
        return other

    def copy(self) -> 'Baffle[ObscurationT]':
        other = super().copy()      # type: Baffle[ObscurationT]
        if self.apertures_base is not None:
            other.apertures_base = self.apertures_base.copy()
        other.apertures_extra = [aper.copy() for aper in self.apertures_extra]
        other.obscuration_base = self.obscuration_base.copy()
        other.margin = self.margin.copy()
        other.min_distance = self.min_distance.copy()
        other.combined_axes = self.combined_axes
        other.shapely_unit = self.shapely_unit
        return other


class BaffleList(
    collections.UserList,
):

    # def concat_apertures_from_raytrace(
    #         self,
    #         raytrace: rays.RaysList,
    #         # lofts: typ.List[typ.List[surface.Surface]],
    # ) -> 'BaffleList':
    #     return BaffleList([b.concat_apertures_from_raytrace(raytrace) for b in self])

    def concat_apertures_from_global_positions(
            self,
            position_1: vector.Vector3D,
            position_2: vector.Vector3D,
            mask: typ.Optional = None,
            hull_axes: typ.Optional[typ.Sequence[int]] = None,
    ) -> 'BaffleList':
        baffle_list = [b.concat_apertures_from_global_positions(position_1, position_2, mask, hull_axes) for b in self]
        return BaffleList(baffle_list)

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[str, str] = ('x', 'y'),
            transform_extra: typ.Optional[tfrm.rigid.TransformList] = None
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()

        if transform_extra is None:
            transform_extra = tfrm.rigid.TransformList()

        for baffle in self:
            baffle.plot(ax=ax, components=components, transform_extra=transform_extra, to_global=True)

        return ax

    def to_dxf(self, file_base: pathlib.Path):
        for i in range(len(self)):
            filename = file_base.parent / (str(file_base.name) + '_' + str(i) + '.dxf')
            self[i].to_dxf(filename)
