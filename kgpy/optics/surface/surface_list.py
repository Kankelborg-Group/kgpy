import typing as typ
import collections
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from kgpy import vector, transform, optics
from . import Surface, CoordinateTransform

__all__ = ['SurfaceList']


class SurfaceList(
    collections.UserList,
    typ.List[Surface],
):

    def raytrace(self, rays: optics.Rays) -> optics.RaysList:
        rays_list = optics.RaysList()
        for surf in self:
            rays = rays.apply_transform(surf.pre_transform.inverse)
            rays = surf.propagate_rays(rays)
            rays_list.append(rays)
            rays = rays.apply_transform(surf.post_transform.inverse)
        return rays_list

    def rays_list_to_global(self, rays_list: optics.RaysList) -> optics.RaysList:
        rays_list_global = optics.RaysList()
        t_global = transform.rigid.TransformList()
        for surf, rays in zip(self, rays_list):
            t_global += surf.pre_transform
            rays_list_global.append(rays.apply_transform(t_global))
            t_global += surf.post_transform
        return rays_list_global

    def plot_2d(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[int, int] = (vector.ix, vector.iy),
            index_first: int = 0,
            index_last: int = ~0,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()

        index_last = index_last % len(self)

        t_global = transform.rigid.TransformList()
        for i, surf in enumerate(self):
            t_global += surf.pre_transform
            if index_first <= i <= index_last:
                surf.plot_2d(ax=ax, rigid_transform=t_global, components=components)
            t_global += surf.post_transform

        return ax
