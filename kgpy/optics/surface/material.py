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
import kgpy.format
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
    name: str = ''

    @abc.abstractmethod
    def index_of_refraction(self, rays: Rays) -> u.Quantity:
        pass

    def transmissivity(self, wavelength: u.Quantity) -> u.Quantity:
        return 100 * u.percent

    def copy(self) -> 'Material':
        other = super().copy()
        other.name = self.name
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
        return []


@dataclasses.dataclass
class Mirror(Material):
    name: str = 'mirror'
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

                plot_kwargs_broadcasted = {}
                for key in plot_kwargs:
                    plot_kwargs_broadcasted[key] = np.broadcast_to(
                        np.array(plot_kwargs[key]), wire.shape[:~0]).reshape(-1)

                for i in range(wire.shape[0]):
                    plot_kwargs_z = {}
                    if component_z is not None:
                        plot_kwargs_z['zs'] = wire[i].get_component(component_z)
                    plot_kwargs_i = {}
                    for key in plot_kwargs_broadcasted:
                        plot_kwargs_i[key] = plot_kwargs_broadcasted[key][i]
                    lines += ax.plot(
                        wire[i].get_component(c1),
                        wire[i].get_component(c2),
                        **plot_kwargs_i,
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

                    plot_kwargs_broadcasted = {}
                    for key in plot_kwargs:
                        plot_kwargs_broadcasted[key] = np.broadcast_to(
                            np.array(plot_kwargs[key])[..., np.newaxis], vertices.shape[:~0]).reshape(-1)

                    vertices = vertices.reshape((-1, ) + vertices.shape[~0:])

                    for i in range(vertices.shape[0]):
                        plot_kwargs_z = {}
                        if component_z is not None:
                            plot_kwargs_z['zs'] = vertices[i].get_component(component_z)
                        plot_kwargs_i = {}
                        for key in plot_kwargs_broadcasted:
                            plot_kwargs_i[key] = plot_kwargs_broadcasted[key][i]
                        lines += ax.plot(
                            vertices[i].get_component(c1),
                            vertices[i].get_component(c2),
                            **plot_kwargs_i,
                            # color=color,
                            # linewidth=linewidth,
                            # linestyle=linestyle,
                            **plot_kwargs_z,
                        )

        return lines


@dataclasses.dataclass
class MultilayerMirror(Mirror):
    layer_material: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    layer_thickness: u.Quantity = dataclasses.field(default_factory=lambda: u.Quantity([]))
    # material_color: typ.Dict[str, str] = dataclasses.field(default_factory=lambda: {})
    # label_x: typ.Dict[str, float] = dataclasses.field(default_factory=lambda: {})

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        if not isinstance(other, MultilayerMirror):
            return False
        if not (self.layer_material == other.layer_material).all():
            return False
        if not (self.layer_thickness == other.layer_thickness).all():
            return False
        # if not self.material_color == other.material_color:
        #     return False
        return True

    def transmissivity(self, wavelength: u.Quantity) -> u.Quantity:
        raise NotImplementedError

    def view(self) -> 'MultilayerMirror':
        other = super().view()  # type: MultilayerMirror
        other.layer_element = self.layer_element
        other.layer_thickness = self.layer_thickness
        return other

    def copy(self) -> 'MultilayerMirror':
        other = super().copy()  # type: MultilayerMirror
        if self.layer_material is not None:
            other.layer_element = self.layer_material.copy()
        if self.layer_thickness is not None:
            other.layer_thickness = self.layer_thickness.copy()
        return other

    def plot_layers(
            self,
            ax: matplotlib.axes.Axes,
            layer_material_color: typ.Dict[str, str],
            layer_label_x: typ.Dict[str, float],
            layer_label_x_text: typ.Dict[str, float],
    ):
        with astropy.visualization.quantity_support():
            z = 0 * u.nm
            for material, thickness in zip(self.layer_material, self.layer_thickness):
                z_new = z + thickness
                ax.axhspan(
                    ymin=z,
                    ymax=z_new,
                    color=layer_material_color[material],
                )
                lx = layer_label_x[material]
                lx_text = layer_label_x_text[material]
                if lx_text >= 1.0:
                    ha = 'left'
                elif lx_text <= 0.0:
                    ha = 'right'
                else:
                    ha = 'center'

                if lx != lx_text:
                    arrowprops=dict(
                        arrowstyle='->',
                    )
                else:
                    arrowprops=None
                z_mid = (z + z_new) / 2
                ax.annotate(
                    text=f'{material} ({kgpy.format.quantity(thickness, digits_after_decimal=0)})',
                    xy=(lx, z_mid),
                    xytext=(lx_text, z_mid),
                    ha=ha,
                    va='center',
                    arrowprops=arrowprops,
                )
                z = z_new

            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)


@dataclasses.dataclass
class AluminumThinFilm(Material):
    name: str = 'thin film Al'
    thickness: u.Quantity = 0 * u.nm
    mesh_ratio: u.Quantity = 100 * u.percent

    def index_of_refraction(self, rays: Rays) -> u.Quantity:
        return 1 * u.dimensionless_unscaled

    def copy(self) -> 'AluminumThinFilm':
        other = super().copy()
        other.thickness = self.thickness.copy()
        other.mesh_ratio = self.mesh_ratio.copy()
        return other
