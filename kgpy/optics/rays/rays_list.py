import typing as typ
import collections
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from kgpy import vector, transform
from . import Rays

__all__ = ['RaysList']


class RaysList(
    collections.UserList,
    typ.List[Rays],
):
    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[int, int] = (vector.ix, vector.iy),
            rigid_transform: typ.Optional[transform.rigid.TransformList] = None,
            color_axis: int = Rays.axis.wavelength,
            plot_vignetted: bool = False,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()

        if rigid_transform is None:
            rigid_transform = transform.rigid.TransformList()

        intercepts = []
        for rays in self:
            rays_transform = rigid_transform + rays.transform
            intercepts.append(rays_transform(rays.position, num_extra_dims=5))
        intercepts = u.Quantity(intercepts)

        img_rays = self[~0]

        color_axis = (color_axis % img_rays.axis.ndim) - img_rays.axis.ndim

        if plot_vignetted:
            mask = img_rays.error_mask
        else:
            mask = img_rays.mask
        mask = np.broadcast_to(mask, img_rays.grid_shape)

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
