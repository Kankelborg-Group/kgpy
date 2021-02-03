import typing as typ
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from kgpy import vector, transform, mixin

__all__ = ['Breadboard']


@dataclasses.dataclass
class Breadboard(
    transform.rigid.Transformable,
    mixin.Colorable,
):
    # num_taps: typ.Tuple[int, int] = (0, 0)
    # tap_diameter: u.Quantity = 0 * u.mm
    # tap_pitch: u.Quantity = 0 * u.mm
    # border_x: u.Quantity = 0 * u.mm
    # border_y: u.Quantity = 0 * u.mm
    color: str = 'gray'
    length: u.Quantity = 0 * u.mm
    width: u.Quantity = 0 * u.mm
    thickness: u.Quantity = 0 * u.mm

    # @property
    # def tap_radius(self):
    #     return self.tap_diameter / 2
    #
    # @property
    # def tap(self) -> u.Quantity:
    #     n_samples = 100
    #     angle = np.linspace(0 * u.deg, 360 * u.deg, num=n_samples)
    #     return vector.from_components(
    #         z=self.tap_radius * np.cos(angle),
    #         x=self.tap_radius * np.sin(angle),
    #     )
    #
    # @property
    # def grid_size(self) -> u.Quantity:
    #     return self.tap_pitch * self.num_taps
    #
    # @property
    # def tap_grid(self):
    #     sz = self.grid_size
    #     return vector.from_components(
    #         z=np.linspace(0, sz[vector.ix], self.num_taps[vector.ix])[None, :],
    #         x=np.linspace(0, sz[vector.iy], self.num_taps[vector.iy])[:, None],
    #     )

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[str, str] = ('x', 'y'),
            color: str = 'black',
            transform_extra: typ.Optional[transform.rigid.TransformList] = None,
            to_global: bool = False,
    ) -> plt.Axes:

        if ax is None:
            _, ax = plt.subplots()

        if color is None:
            color = self.color

        if to_global:
            if transform_extra is None:
                transform_extra = transform.rigid.TransformList()
            transform_extra = transform_extra + self.transform

        top = vector.Vector3D(
            x=u.Quantity([0 * u.mm, 0 * u.mm, self.width, self.width]),
            y=0 * u.mm,
            z=u.Quantity([0 * u.mm, self.length, self.length, 0 * u.mm]),
        )
        bottom = top.copy()
        bottom.y = -self.thickness
        points = np.stack([top, bottom])
        points = transform_extra(points)

        c0, c1 = components
        ax.fill(points.get_component(c0).T, points.get_component(c1).T, fill=False, color=color)
        ax.plot(points.get_component(c0), points.get_component(c1), color=color)

        return ax


