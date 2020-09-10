import typing as typ
import collections
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from kgpy import vector, transform, optics
from . import Surface

__all__ = ['SurfaceList']


class SurfaceList(
    collections.UserList,
    typ.List[Surface],
):

    def raytrace(self, rays: optics.Rays, intercept_error: u.Quantity = 0.1 * u.nm) -> optics.RaysList:
        rays_list = optics.RaysList()
        for surf in self:
            rays = surf.propagate_rays(rays, intercept_error=intercept_error)
            rays_list.append(rays)
        return rays_list

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[int, int] = (vector.ix, vector.iy),
            rigid_transform: typ.Optional[transform.rigid.TransformList] = None
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()

        if rigid_transform is None:
            rigid_transform = transform.rigid.TransformList()

        for surf in self:
            surf.plot(ax=ax, components=components, rigid_transform=rigid_transform + surf.transform)

        return ax
