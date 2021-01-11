import abc
import dataclasses
import typing as typ
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
from kgpy import mixin, vector, transform
from ..rays import Rays
from .aperture import Aperture, Polygon

__all__ = ['Material', 'Mirror']


@dataclasses.dataclass
class Material(
    mixin.Broadcastable,
    mixin.Copyable,
    abc.ABC
):

    @abc.abstractmethod
    def index_of_refraction(self, rays: Rays) -> u.Quantity:
        pass

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[int, int] = (vector.ix, vector.iy),
            rigid_transform: typ.Optional[transform.rigid.TransformList] = None,
            sag: typ.Optional[typ.Callable[[u.Quantity, u.Quantity], u.Quantity]] = None,
            aperture: typ.Optional[Aperture] = None,
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        return ax


@dataclasses.dataclass
class Mirror(Material):
    thickness: u.Quantity = 0 * u.mm

    def index_of_refraction(self, rays: Rays) -> u.Quantity:
        return -np.sign(rays.index_of_refraction) * u.dimensionless_unscaled

    def copy(self) -> 'Mirror':
        other = super().copy()      # type: Mirror
        other.thickness = self.thickness

    def plot(
            self,
            ax: typ.Optional[plt.Axes] = None,
            components: typ.Tuple[int, int] = (vector.ix, vector.iy),
            rigid_transform: typ.Optional[transform.rigid.TransformList] = None,
            sag: typ.Optional[typ.Callable[[u.Quantity, u.Quantity], u.Quantity]] = None,
            aperture: typ.Optional[Aperture] = None,
    ) -> plt.Axes:
        super().plot(ax=ax, components=components, rigid_transform=rigid_transform, sag=sag, aperture=aperture)

        if aperture is not None:
            with astropy.visualization.quantity_support():

                c1, c2 = components
                wire = aperture.wire.copy()
                wire[vector.z] = self.thickness
                if rigid_transform is not None:
                    wire = rigid_transform(wire, num_extra_dims=1)
                wire = wire.reshape((-1,) + wire.shape[~1:])
                ax.fill(wire[..., c1].T, wire[..., c2].T, fill=False)

                # todo: utilize polymorphsim here
                if isinstance(aperture, Polygon):

                    front_vertices = aperture.vertices.copy()
                    back_vertices = aperture.vertices.copy()
                    front_vertices[vector.z] = sag(front_vertices[vector.x], front_vertices[vector.y])
                    back_vertices[vector.z] = self.thickness

                    vertices = np.stack([front_vertices, back_vertices], axis=~1)
                    if rigid_transform is not None:
                        vertices = rigid_transform(vertices, num_extra_dims=2)
                    vertices = vertices.reshape((-1, ) + vertices.shape[~1:])

                    ax.plot(vertices[..., c1].T, vertices[..., c2].T, color='black')

        return ax