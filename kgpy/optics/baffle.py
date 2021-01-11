import typing as typ
import abc
import collections
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import shapely.geometry
import shapely.ops
from kgpy import mixin, vector, transform as tfrm, geometry, optics
from . import rays, surface

__all__ = ['Baffle', 'BaffleList']

ApertureT = typ.TypeVar('ApertureT', bound=typ.List[surface.Aperture])
ObscurationT = typ.TypeVar('ObscurationT', bound=surface.Aperture)


@dataclasses.dataclass
class Baffle(
    mixin.Broadcastable,
    tfrm.rigid.Transformable,
    mixin.Named,
    abc.ABC,
    typ.Generic[ObscurationT],
):

    apertures: typ.Optional[ApertureT] = None
    obscuration: typ.Optional[ObscurationT] = None
    margin: u.Quantity = 1 * u.mm
    union_axes: typ.Optional[typ.Sequence[int]] = None

    def apertures_from_raytrace(
            self,
            surfaces: surface.SurfaceList,
            raytrace: rays.RaysList,
    ) -> 'Baffle':
        baffle_plane_point = self.transform.translation_eff
        baffle_plane_normal = self.transform(vector.z_hat, translate=False)

        img_rays = raytrace[~0]
        sh = img_rays.shape
        vgrid_sh = img_rays.vector_grid_shape

        convex_hull_list = []

        position = []
        for rays in raytrace:
            t = self.transform.inverse + rays.transform
            from_prev_to_self = rays.transform.inverse + self.transform
            p = t(rays.position, num_extra_dims=5)
            p = np.broadcast_to(p, vgrid_sh, subok=True)
            position.append(p)
        position = u.Quantity(position)

        for i in range(1, len(raytrace)):

            position_1 = position[i - 1]
            position_2 = position[i]

            intercept = geometry.segment_plane_intercept(
                plane_point=[0, 0, 0] * u.mm,
                plane_normal=vector.z_hat,
                line_point_1=position_1,
                line_point_2=position_2,
            )

            mask = img_rays.mask
            mask &= np.isfinite(intercept.sum(~0))
            mask &= ~self.obscuration.is_unvignetted(intercept)

            surf = surfaces[i]
            if surf.baffle_link is not None:
                t1 = self.transform.inverse + surf.transform
                t2 = self.transform.inverse + surf.baffle_link.transform
                p1 = t1(surf.aperture.vertices, num_extra_dims=1)[..., :, None, :]
                p2 = t2(surf.baffle_link.aperture.vertices, num_extra_dims=1)[..., None, :, :]

                intercept_surf = geometry.segment_plane_intercept(
                    plane_point=[0, 0, 0] * u.mm,
                    plane_normal=vector.z_hat,
                    line_point_1=p1,
                    line_point_2=p2,
                )
            else:
                intercept_surf = None

            if self.union_axes is None:
                intercept = intercept[None, mask, :]
                if intercept_surf is not None:
                    intercept_surf = intercept_surf[None, np.isfinite(intercept_surf.sum(~0)), :]
                    # intercept_surf = intercept_surf.reshape((1, -1, 3))
                # position_1 = position_1[None, mask, :]
                # position_2 = position_2[None, mask, :]

            else:
                raise NotImplementedError
                # position_1 = position_1.reshape(sh + (-1, 3))
                # position_2 = position_2.reshape(sh + (-1, 3))
                #
                # num_union_axes = len(self.union_axes)
                # num_separate_axes = len(sh) - num_union_axes
                # union_axes_dest = num_separate_axes + np.arange(num_union_axes)
                #
                # position_1 = np.moveaxis(position_1, self.union_axes, union_axes_dest)
                # position_2 = np.moveaxis(position_2, self.union_axes, union_axes_dest)
                #
                # position_1 = position_1.reshape(position_1.shape[:num_separate_axes] + (-1, 3))
                # position_2 = position_2.reshape(position_2.shape[:num_separate_axes] + (-1, 3))
                #
                # position_1 = position_1.reshape((-1, ) + position_1.shape[~1:])
                # position_2 = position_2.reshape((-1, ) + position_2.shape[~1:])

            for j in range(len(intercept)):

                # intercept = geometry.segment_plane_intercept(
                #     plane_point=baffle_plane_point,
                #     plane_normal=baffle_plane_normal,
                #     line_point_1=position_1[j],
                #     line_point_2=position_2[j],
                # )
                #
                # intercept = intercept[np.isfinite(intercept.sum(~0))]
                # print(intercept.shape)
                #
                # intercept = intercept[~self.obscuration.is_unvignetted(intercept), :]

                intercept_j = intercept[j]
                if intercept_surf is not None:
                    intercept_j = np.concatenate([intercept_j, intercept_surf[j]])

                convex_hull = shapely.geometry.MultiPoint(intercept_j.value).convex_hull
                convex_hull_list.append(convex_hull)


        apertures = shapely.ops.unary_union(convex_hull_list)
        if isinstance(apertures, shapely.geometry.Polygon):
            apertures = shapely.geometry.MultiPolygon([apertures])

        apertures = shapely.geometry.MultiPolygon([a.buffer(self.margin.to(position.unit).value) for a in apertures])
        apertures = shapely.ops.unary_union(apertures)
        if isinstance(apertures, shapely.geometry.Polygon):
            apertures = shapely.geometry.MultiPolygon([apertures])

        # num_dilate_iter = 3
        # for d in range(num_dilate_iter):
        #     apertures = shapely.geometry.MultiPolygon([aper.buffer(self.margin.value / num_dilate_iter) for aper in apertures])
        #     apertures = shapely.ops.unary_union(apertures)
        #     if isinstance(apertures, shapely.geometry.Polygon):
        #         apertures = shapely.geometry.MultiPolygon([apertures])

        apertures_final = []
        for aper in apertures:
            a = optics.surface.aperture.IrregularPolygon(vertices=vector.to_3d(np.array(aper.exterior) << position.unit))
            apertures_final.append(a)

        return Baffle(
            name=self.name.copy(),
            transform=self.transform.copy(),
            apertures=apertures_final,
            obscuration=self.obscuration.copy(),
        )

    def unary_union(self, other: 'Baffle'):

        aper_unit = u.mm

        apertures_self = [shapely.geometry.Polygon(aper.vertices.to(aper_unit).value) for aper in self.apertures]
        apertures_other = [shapely.geometry.Polygon(aper.vertices.to(aper_unit).value) for aper in other.apertures]

        apertures = shapely.geometry.MultiPolygon(apertures_self + apertures_other)
        apertures = shapely.ops.unary_union(apertures)
        if isinstance(apertures, shapely.geometry.Polygon):
            apertures = shapely.geometry.MultiPolygon([apertures])

        apertures_final = []
        for aper in apertures:
            a = optics.surface.aperture.IrregularPolygon(
                vertices=vector.to_3d(np.array(aper.exterior) << aper_unit))
            apertures_final.append(a)

        return Baffle(
            name=self.name.copy(),
            transform=self.transform.copy(),
            apertures=apertures_final,
            obscuration=self.obscuration.copy(),
        )

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[int, int] = (0, 1),
            rigid_transform: typ.Optional[tfrm.rigid.TransformList] = None,
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()

        if self.apertures is not None:
            for aper in self.apertures:
                aper.plot(ax=ax, components=components, rigid_transform=rigid_transform)

        if self.obscuration is not None:
            self.obscuration.plot(ax=ax, components=components, rigid_transform=rigid_transform)

        return ax

    def copy(self) -> 'Baffle[ApertureT, ObscurationT]':
        other = super().copy()      # type: Baffle[ApertureT, ObscurationT]
        other.apertures = [aper.copy() for aper in self.apertures]
        other.obscuration = self.obscuration.copy()
        return other


class BaffleList(
    collections.UserList,
):

    def apertures_from_raytrace(
            self,
            surfaces: surface.SurfaceList,
            raytrace: rays.RaysList,
    ) -> 'BaffleList':
        data = []
        for baffle in self:
            baffle = baffle.apertures_from_raytrace(surfaces, raytrace)
            data.append(baffle)
        return BaffleList(data)


    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[int, int] = (vector.ix, vector.iy),
            rigid_transform: typ.Optional[tfrm.rigid.TransformList] = None
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()

        if rigid_transform is None:
            rigid_transform = tfrm.rigid.TransformList()

        for baffle in self:
            baffle.plot(ax, components, rigid_transform + baffle.transform)

        return ax
