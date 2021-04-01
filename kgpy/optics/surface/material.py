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
    mixin.Plottable,
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
            plot_kwargs: typ.Optional[typ.Dict[str, typ.Any]] = None,
            # color: typ.Optional[str] = None,
            # linewidth: typ.Optional[float] = None,
            # linestyle: typ.Optional[str] = None,
            transform_extra: typ.Optional[transform.rigid.TransformList] = None,
            sag: typ.Optional[typ.Callable[[u.Quantity, u.Quantity], u.Quantity]] = None,
            aperture: typ.Optional[Aperture] = None,
    ) -> typ.List[matplotlib.lines.Line2D]:
        return []


@dataclasses.dataclass
class Mirror(Material):
    thickness: typ.Optional[u.Quantity] = None

    def index_of_refraction(self, rays: Rays) -> u.Quantity:
        return -np.sign(rays.index_of_refraction) * u.dimensionless_unscaled

    def view(self) -> 'Mirror':
        other = super().view()      # type: Mirror
        other.thickness = self.thickness
        return other

    def copy(self) -> 'Mirror':
        other = super().copy()      # type: Mirror
        if self.thickness is not None:
            other.thickness = self.thickness.copy()
        else:
            other.thickness = self.thickness
        return other

    def plot(
            self,
            ax: matplotlib.axes.Axes,
            components: typ.Tuple[str, str] = ('x', 'y'),
            component_z: typ.Optional[str] = None,
            plot_kwargs: typ.Optional[typ.Dict[str, typ.Any]] = None,
            # color: typ.Optional[str] = None,
            # linewidth: typ.Optional[float] = None,
            # linestyle: typ.Optional[str] = None,
            transform_extra: typ.Optional[transform.rigid.TransformList] = None,
            sag: typ.Optional[typ.Callable[[u.Quantity, u.Quantity], u.Quantity]] = None,
            aperture: typ.Optional[Aperture] = None,
    ) -> typ.List[matplotlib.lines.Line2D]:

        if plot_kwargs is not None:
            plot_kwargs = {**self.plot_kwargs, **plot_kwargs}
        else:
            plot_kwargs = self.plot_kwargs

        # if color is None:
        #     color = self.color
        # if linewidth is None:
        #     linewidth = self.linewidth
        # if linestyle is None:
        #     linestyle = self.linestyle

        lines = []
        lines += super().plot(
            ax=ax,
            components=components,
            plot_kwargs=plot_kwargs,
            # color=color,
            transform_extra=transform_extra,
            sag=sag,
            aperture=aperture
        )

        if aperture is not None and self.thickness is not None:
            with astropy.visualization.quantity_support():

                c1, c2 = components
                wire = aperture.wire.copy()
                wire.z = self.thickness
                if transform_extra is not None:
                    wire = transform_extra(wire, num_extra_dims=1)
                wire = wire.reshape((-1,) + wire.shape[~0:])

                for i in range(wire.shape[0]):
                    plot_kwargs_z = {}
                    if component_z is not None:
                        plot_kwargs_z['zs'] = wire[i].get_component(component_z)
                    lines += ax.plot(
                        wire[i].get_component(c1),
                        wire[i].get_component(c2),
                        **plot_kwargs,
                        # color=color,
                        # linewidth=linewidth,
                        # linestyle=linestyle,
                        **plot_kwargs_z,
                    )

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

                    for i in range(vertices.shape[0]):
                        plot_kwargs_z = {}
                        if component_z is not None:
                            plot_kwargs_z['zs'] = vertices[i].get_component(component_z)

                        lines += ax.plot(
                            vertices[i].get_component(c1),
                            vertices[i].get_component(c2),
                            **plot_kwargs,
                            # color=color,
                            # linewidth=linewidth,
                            # linestyle=linestyle,
                            **plot_kwargs_z,
                        )

        return lines
