import abc
import dataclasses
import pathlib
import typing as typ
import numpy as np
import pandas
import scipy.interpolate
import matplotlib.axes
import matplotlib.lines
import matplotlib.patches
import mpl_toolkits.mplot3d.art3d
import astropy.units as u
import astropy.constants
import astropy.visualization
import xrt.backends.raycing.materials
import thermo
from kgpy import mixin, vector, transform
import kgpy.format
import kgpy.plot
from ...rays import Rays
from ..aperture import Aperture, Polygon

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

    def transmissivity(self, rays: Rays) -> u.Quantity:
        return 1 * u.dimensionless_unscaled

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
                    plot_kwargs_i = {}
                    for key in plot_kwargs_broadcasted:
                        plot_kwargs_i[key] = plot_kwargs_broadcasted[key][i]
                    if component_z is None:
                        lines += ax.plot(
                            wire[i].get_component(c1),
                            wire[i].get_component(c2),
                            **plot_kwargs_i,
                            # color=color,
                            # linewidth=linewidth,
                            # linestyle=linestyle,
                        )
                    else:
                        if 'color' in plot_kwargs_i:
                            plot_kwargs_i['edgecolors'] = plot_kwargs_i.pop('color')
                        if 'linewidth' in plot_kwargs_i:
                            plot_kwargs_i['linewidths'] = plot_kwargs_i.pop('linewidth')
                        if 'linestyle' in plot_kwargs_i:
                            plot_kwargs_i['linestyles'] = plot_kwargs_i.pop('linestyle')

                        lines += [ax.add_collection(mpl_toolkits.mplot3d.art3d.Poly3DCollection(
                            verts=[np.stack([
                                wire[i].get_component(components[0]),
                                wire[i].get_component(components[1]),
                                wire[i].get_component(component_z),
                            ], axis=~0)],
                            facecolors='white',
                            **plot_kwargs_i
                        ))]

                # todo: utilize polymorphsim here
                if isinstance(aperture, Polygon):

                    if component_z is None:

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

                        vertices = vertices.reshape((-1,) + vertices.shape[~0:])

                        for i in range(vertices.shape[0]):
                            plot_kwargs_i = {}
                            for key in plot_kwargs_broadcasted:
                                plot_kwargs_i[key] = plot_kwargs_broadcasted[key][i]

                            lines += ax.plot(
                                vertices[i].get_component(c1),
                                vertices[i].get_component(c2),
                                **plot_kwargs_i,
                            )

                    else:
                        for v in range(1, aperture.vertices.shape[~0]):
                            vert_left = aperture.vertices[..., v - 1]
                            vert_right = aperture.vertices[..., v]
                            vert_diff = vert_right - vert_left
                            t = np.linspace(0, 1, num=aperture.num_samples, endpoint=True)
                            wire_top = vert_left[..., np.newaxis] + vert_diff * t
                            wire_top.z = wire_top.z + sag(wire_top.x, wire_top.y)
                            wire_bottom = wire_top.copy()[..., ::-1]
                            wire_bottom.z = self.thickness
                            wire = np.concatenate([wire_top, wire_bottom], axis=~0)
                            if transform_extra is not None:
                                wire = transform_extra(wire, num_extra_dims=1)

                            plot_kwargs_broadcasted = {}
                            for key in plot_kwargs:
                                plot_kwargs_broadcasted[key] = np.broadcast_to(
                                    np.array(plot_kwargs[key]), wire.shape[:~0]).reshape(-1)

                            wire = wire.reshape(-1, wire.shape[~0])
                            for i in range(wire.shape[0]):
                                plot_kwargs_i = {}
                                for key in plot_kwargs_broadcasted:
                                    plot_kwargs_i[key] = plot_kwargs_broadcasted[key][i]
                                if 'color' in plot_kwargs_i:
                                    plot_kwargs_i['edgecolors'] = plot_kwargs_i.pop('color')
                                if 'linewidth' in plot_kwargs_i:
                                    plot_kwargs_i['linewidths'] = plot_kwargs_i.pop('linewidth')
                                if 'linestyle' in plot_kwargs_i:
                                    plot_kwargs_i['linestyles'] = plot_kwargs_i.pop('linestyle')

                                lines += [ax.add_collection(mpl_toolkits.mplot3d.art3d.Poly3DCollection(
                                    verts=[np.stack([
                                        wire[i].get_component(components[0]),
                                        wire[i].get_component(components[1]),
                                        wire[i].get_component(component_z),
                                    ], axis=~0)],
                                    facecolors='white',
                                    **plot_kwargs_i
                                ))]

        return lines


@dataclasses.dataclass
class Layer(mixin.Copyable):
    material: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    thickness: u.Quantity = dataclasses.field(default_factory=lambda: u.Quantity([]))
    num_periods: int = 1

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        if not isinstance(other, Layer):
            return False
        if not (self.material == other.material).all():
            return False
        if not (self.thickness == other.thickness).all():
            return False
        if not (self.num_periods == other.num_periods):
            return False
        return True

    def plot(
            self,
            ax: matplotlib.axes.Axes,
            z: u.Quantity,
            layer_material_color: typ.Dict[str, str],
            layer_label_x: typ.Dict[str, float],
            layer_label_x_text: typ.Dict[str, float],
    ) -> u.Quantity:
        with astropy.visualization.quantity_support():

            x_substrate = np.linspace(0, 1, 1000)
            substrate_amplitude = 5 * u.nm
            y_substrate = substrate_amplitude * np.sin(2 * np.pi * x_substrate)
            substrate_thickness = 20 * u.nm
            ax.fill_between(
                x=x_substrate,
                y1=0,
                y2=-(y_substrate + substrate_thickness),
                facecolor='gray',
                edgecolor='none',
            )
            ax.text(
                x=0.5,
                y=-(substrate_thickness - substrate_amplitude)/2,
                s='substrate',
                va='center',
                ha='center',
            )

            z_start = z
            for material, thickness in zip(self.material, self.thickness):
                z_new = z + thickness
                ax.add_patch(matplotlib.patches.Rectangle(
                    xy=(0, z),
                    width=1,
                    height=thickness,
                    facecolor=layer_material_color[material],
                    edgecolor='none',
                ))
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

            ax.autoscale_view()

            if self.num_periods > 1:
                kgpy.plot.brace.vertical(
                    ax=ax,
                    x=0,
                    ymin=z_start,
                    ymax=z_new,
                    width=-0.1,
                    text=f'$\\times${self.num_periods}',
                    beta=3 / u.nm,
                )

        return z_new


@dataclasses.dataclass
class MultilayerMirror(Mirror):
    cap: Layer = dataclasses.field(default_factory=Layer)
    main: Layer = dataclasses.field(default_factory=Layer)
    base: Layer = dataclasses.field(default_factory=Layer)

    # def transmissivity(self, rays: Rays) -> u.Quantity:
    #     raise NotImplementedError

    def plot_layers(
            self,
            ax: matplotlib.axes.Axes,
            layer_material_color: typ.Dict[str, str],
            layer_label_x: typ.Dict[str, float],
            layer_label_x_text: typ.Dict[str, float],
    ):
        z = 0 * u.nm
        z = self.base.plot(
            ax=ax,
            z=z,
            layer_material_color=layer_material_color,
            layer_label_x=layer_label_x,
            layer_label_x_text=layer_label_x_text,
        )
        z = self.main.plot(
            ax=ax,
            z=z,
            layer_material_color=layer_material_color,
            layer_label_x=layer_label_x,
            layer_label_x_text=layer_label_x_text,
        )
        z = self.cap.plot(
            ax=ax,
            z=z,
            layer_material_color=layer_material_color,
            layer_label_x=layer_label_x,
            layer_label_x_text=layer_label_x_text,
        )


@dataclasses.dataclass
class MeasuredMultilayerMirror(MultilayerMirror):
    efficiency_data: typ.Optional[u.Quantity] = None
    wavelength_data: typ.Optional[u.Quantity] = None

    def transmissivity(self, rays: Rays) -> u.Quantity:
        interp = scipy.interpolate.interp1d(self.wavelength_data, self.efficiency_data, bounds_error=False)
        return interp(rays.wavelength.to(self.wavelength_data.unit)) * self.efficiency_data.unit

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        if not (self.efficiency_data == other.efficiency_data).all():
            return False
        if not (self.wavelength_data == other.wavelength_data).all():
            return False
        return True


@dataclasses.dataclass
class AluminumThinFilm(Material):
    name: str = 'Al'
    thickness: u.Quantity = 0 * u.nm
    thickness_oxide: u.Quantity = 0 * u.nm
    mesh_ratio: u.Quantity = 100 * u.percent
    mesh_material: str = ''
    density_ratio: float = 0.9
    xrt_table: str = 'Henke'

    @property
    def xrt_aluminum(self) -> xrt.backends.raycing.materials.Material:
        return xrt.backends.raycing.materials.Material(
            elements='Al',
            kind='plate',
            t=self.thickness.to(u.mm).value,
            table=self.xrt_table,
            rho=self.density_ratio * (thermo.Chemical('Al').rho * u.kg / u.m ** 3).to(u.g / u.cm ** 3).value,
        )

    @property
    def xrt_aluminum_oxide(self) -> xrt.backends.raycing.materials.Material:
        return xrt.backends.raycing.materials.Material(
            elements=['Al', 'O', ],
            quantities=[2, 3],
            kind='plate',
            t=self.thickness_oxide.to(u.mm).value,
            table=self.xrt_table,
            rho=self.density_ratio * (thermo.Chemical('Al2O3').rho * u.kg / u.m ** 3).to(u.g / u.cm ** 3).value,
        )

    def transmissivity_aluminum(self, rays: Rays) -> u.Quantity:
        absorption = self.xrt_aluminum.get_absorption_coefficient(rays.energy.to(u.eV).value) / u.cm
        transmissivity = np.exp(-absorption * self.thickness / rays.direction.z)
        return transmissivity

    def transmissivity_aluminum_oxide(self, rays: Rays) -> u.Quantity:
        absorption = self.xrt_aluminum_oxide.get_absorption_coefficient(rays.energy.to(u.eV).value) / u.cm
        transmissivity = np.exp(-absorption * 2 * self.thickness_oxide / rays.direction.z)
        return transmissivity

    def transmissivity(self, rays: Rays) -> u.Quantity:
        mesh_ratio = self.mesh_ratio.to(u.dimensionless_unscaled)
        return mesh_ratio * self.transmissivity_aluminum(rays) * self.transmissivity_aluminum_oxide(rays)

    def index_of_refraction(self, rays: Rays) -> u.Quantity:
        return 1 * u.dimensionless_unscaled


@dataclasses.dataclass
class CCDStern1994(Material):

    quantum_efficiency_data: typ.ClassVar[u.Quantity] = [
        0.91,
        0.80,
        0.48,
        0.32,
        0.42,
        0.86,
        0.82,
        0.60,
        0.58,
        0.53,
        0.30,
        0.085,
        0.055,
        0.06,
        0.13,
        0.09,
        0.33,
        0.29,
        0.50,
        0.53,
        0.62,
        0.63,
        0.65,
        0.65,
        0.65,
        0.61,
        0.47,
        0.33,
        0.21,
        0.19,
    ] * u.dimensionless_unscaled

    wavelength_data: typ.ClassVar[u.Quantity] = [
        13.3,
        23.6,
        44.7,
        67.6,
        114.0,
        135.5,
        171.4,
        256.0,
        303.8,
        461.0,
        584.0,
        736.0,
        1215.5,
        2537.0,
        3500.0,
        3650.0,
        4000.0,
        4050.0,
        4500.0,
        5000.0,
        5500.0,
        6000.0,
        6500.0,
        7000.0,
        7500.0,
        8000.0,
        8500.0,
        9000.0,
        9500.0,
        10000.0,
    ] * u.AA

    def transmissivity(self, rays: Rays) -> u.Quantity:
        qe_interp = scipy.interpolate.interp1d(self.wavelength_data, self.quantum_efficiency_data)
        qe = qe_interp(rays.wavelength.to(self.wavelength_data.unit)) * self.quantum_efficiency_data.unit
        return qe

    def index_of_refraction(self, rays: Rays) -> u.Quantity:
        return 1 * u.dimensionless_unscaled


@dataclasses.dataclass
class CCDStern2004(Material):

    data_path: typ.ClassVar[pathlib.Path] = pathlib.Path(__file__).parent / 'stern_2004_model_1.50e05.csv'

    @property
    def dataframe(self) -> pandas.DataFrame:
        return pandas.read_csv(self.data_path, header=None)

    @property
    def quantum_efficiency_data(self):
        return self.dataframe[1].to_numpy() * u.dimensionless_unscaled

    @property
    def wavelength_data(self):
        return self.dataframe[0].to_numpy() * u.AA

    def transmissivity(self, rays: Rays) -> u.Quantity:
        qe_interp = scipy.interpolate.interp1d(self.wavelength_data, self.quantum_efficiency_data)
        qe = qe_interp(rays.wavelength.to(self.wavelength_data.unit)) * self.quantum_efficiency_data.unit
        return qe

    def index_of_refraction(self, rays: Rays) -> u.Quantity:
        return 1 * u.dimensionless_unscaled
