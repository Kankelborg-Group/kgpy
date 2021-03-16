import abc
import dataclasses
import typing as typ
import numpy as np
import matplotlib.axes
import matplotlib.lines
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
            ax: matplotlib.axes.Axes,
            components: typ.Tuple[str, str] = ('x', 'y'),
            component_z: typ.Optional[str] = None,
            color: typ.Optional[str] = None,
            transform_extra: typ.Optional[transform.rigid.TransformList] = None,
            sag: typ.Optional[typ.Callable[[u.Quantity, u.Quantity], u.Quantity]] = None,
            aperture: typ.Optional[Aperture] = None,
    ) -> typ.List[matplotlib.lines.Line2D]:
        return []


@dataclasses.dataclass
class Mirror(Material):
    thickness: u.Quantity = 0 * u.mm

    def index_of_refraction(self, rays: Rays) -> u.Quantity:
        return -np.sign(rays.index_of_refraction) * u.dimensionless_unscaled

    def view(self) -> 'Mirror':
        other = super().view()      # type: Mirror
        other.thickness = self.thickness
        return other

    def copy(self) -> 'Mirror':
        other = super().copy()      # type: Mirror
        other.thickness = self.thickness.copy()
        return other

    def plot(
            self,
            ax: matplotlib.axes.Axes,
            components: typ.Tuple[str, str] = ('x', 'y'),
            component_z: typ.Optional[str] = None,
            color: typ.Optional[str] = None,
            transform_extra: typ.Optional[transform.rigid.TransformList] = None,
            sag: typ.Optional[typ.Callable[[u.Quantity, u.Quantity], u.Quantity]] = None,
            aperture: typ.Optional[Aperture] = None,
    ) -> typ.List[matplotlib.lines.Line2D]:
        lines = []
        lines += super().plot(
            ax=ax,
            components=components,
            color=color,
            transform_extra=transform_extra,
            sag=sag,
            aperture=aperture
        )

        if aperture is not None:
            with astropy.visualization.quantity_support():

                c1, c2 = components
                wire = aperture.wire.copy()
                wire.z = self.thickness
                if transform_extra is not None:
                    wire = transform_extra(wire, num_extra_dims=1)
                wire = wire.reshape((-1,) + wire.shape[~0:])
                lines += ax.plot(wire.get_component(c1).T, wire.get_component(c2).T, color=color)

                # todo: utilize polymorphsim here
                if isinstance(aperture, Polygon):

                    front_vertices = aperture.vertices.copy()
                    back_vertices = aperture.vertices.copy()
                    front_vertices.z = sag(front_vertices.x, front_vertices.y)
                    back_vertices.z = self.thickness

                    vertices = np.stack([front_vertices, back_vertices], axis=~0)
                    if transform_extra is not None:
                        vertices = transform_extra(vertices, num_extra_dims=2)

                    vertices = vertices.reshape((-1, ) + vertices.shape[~0:])

                    plot_kwargs_z = {}
                    if component_z is not None:
                        plot_kwargs_z['zs'] = vertices.get_component(component_z).T

                    lines += ax.plot(vertices.get_component(c1).T, vertices.get_component(c2).T, color=color)

        return lines
