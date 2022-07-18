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
import kgpy.mixin
import kgpy.labeled
import kgpy.uncertainty
import kgpy.function
import kgpy.transforms
import kgpy.format
import kgpy.plot
from ... import vectors
from ... import rays
from .. import apertures
from .. import sags

__all__ = [
    'Material',
    'Mirror',
    'Layer',
    'MultilayerMirror',
    'MeasuredMultilayerMirror',
    'AluminumThinFilm',
    'CCDStern1994',
    'CCDStern2004',
]

MaterialT = typ.TypeVar('MaterialT', bound='Material')
MirrorT = typ.TypeVar('MirrorT', bound='Mirror')
LayerT = typ.TypeVar('LayerT', bound='Layer')
MultilayerMirrorT = typ.TypeVar('MultilayerMirrorT', bound='MultilayerMirror')
MeasuredMultilayerMirrorT = typ.TypeVar('MeasuredMultilayerMirrorT', bound='MeasuredMultilayerMirror')
AluminumThinFilmT = typ.TypeVar('AluminumThinFilmT', bound='AluminumThinFilm')
CCDStern1994T = typ.TypeVar('CCDStern1994T', bound='CCDStern1994')
CCDStern2004T = typ.TypeVar('CCDStern2004T', bound='CCDStern2004')


@dataclasses.dataclass
class Material(
    kgpy.mixin.Plottable,
    kgpy.mixin.Broadcastable,
    kgpy.mixin.Copyable,
    abc.ABC
):
    name: str = ''

    @abc.abstractmethod
    def index_refraction(self: MaterialT, ray: rays.RayVector) -> kgpy.uncertainty.ArrayLike:
        pass

    def transmissivity(self: MaterialT, ray: rays.RayVector) -> kgpy.uncertainty.ArrayLike:
        return 1 * u.dimensionless_unscaled

    def plot(
            self: MaterialT,
            ax: matplotlib.axes.Axes,
            component_x: str = 'x',
            component_y: str = 'y',
            component_z: typ.Optional[str] = None,
            transform_extra: typ.Optional[kgpy.transforms.AbstractTransform] = None,
            sag: typ.Optional[sags.Sag] = None,
            aperture: typ.Optional[apertures.Aperture] = None,
            **kwargs: typ.Any,
    ) -> typ.List[matplotlib.lines.Line2D]:
        return []


@dataclasses.dataclass
class Mirror(Material):
    name: str = 'mirror'
    thickness: typ.Optional[kgpy.uncertainty.ArrayLike] = None

    def index_refraction(self: MirrorT, ray: rays.RayVector) -> kgpy.uncertainty.ArrayLike:
        return -np.sign(ray.index_refraction) * u.dimensionless_unscaled

    def plot(
            self: MirrorT,
            ax: matplotlib.axes.Axes,
            component_x: str = 'x',
            component_y: str = 'y',
            component_z: typ.Optional[str] = None,
            transform_extra: typ.Optional[kgpy.transforms.TransformList] = None,
            sag: typ.Optional[sags.Sag] = None,
            aperture: typ.Optional[apertures.Aperture] = None,
            **kwargs,
    ) -> typ.List[matplotlib.lines.Line2D]:

        kwargs = {**self.plot_kwargs, **kwargs}

        result = []
        result += super().plot(
            ax=ax,
            # components=components,
            component_x=component_x,
            component_y=component_y,
            component_z=component_z,
            # color=color,
            transform_extra=transform_extra,
            sag=sag,
            aperture=aperture,
            **kwargs,
        )

        if aperture is not None and self.thickness is not None:

            wire = aperture.wire.copy()
            wire.z = -self.thickness
            if transform_extra is not None:
                wire = transform_extra(wire)

            if isinstance(ax, mpl_toolkits.mplot3d.Axes3D):
                result += wire.plot_filled(
                    ax=ax,
                    axis_plot='wire',
                    component_x=component_x,
                    component_y=component_y,
                    component_z=component_z,
                    **kwargs,
                )
            else:
                result += wire.plot(
                    ax=ax,
                    axis_plot='wire',
                    component_x=component_x,
                    component_y=component_y,
                    component_z=component_z,
                    **kwargs,
                )

            if aperture.vertices is not None:

                if isinstance(ax, mpl_toolkits.mplot3d.Axes3D):
                    for v in range(0, aperture.vertices.shape['vertex']):
                        vert_left = aperture.vertices[dict(vertex=v - 1)]
                        vert_right = aperture.vertices[dict(vertex=v)]
                        vert_diff = vert_right - vert_left
                        t = kgpy.labeled.LinearSpace(0, 1, num=aperture.num_samples, endpoint=True, axis='wire')
                        wire_top = vert_left + vert_diff * t
                        wire_top.z = wire_top.z + sag(wire_top)
                        wire_bottom = wire_top.copy()[dict(wire=slice(None, None, -1))]
                        wire_bottom.z = -self.thickness
                        wire = np.concatenate([wire_top, wire_bottom], axis='wire')
                        if transform_extra is not None:
                            wire = transform_extra(wire)

                        wire.plot_filled(
                            ax=ax,
                            axis_plot='wire',
                            component_x=component_x,
                            component_y=component_y,
                            component_z=component_z,
                            **kwargs,
                        )
                else:
                    front_vertices = aperture.vertices.copy()
                    back_vertices = aperture.vertices.copy()
                    front_vertices.z = sag(front_vertices)
                    back_vertices.z = -self.thickness

                    vertices = np.stack([front_vertices, back_vertices], axis='wire')
                    if transform_extra is not None:
                        vertices = transform_extra(vertices)

                    vertices.plot(
                        ax=ax,
                        axis_plot='wire',
                        component_x=component_x,
                        component_y=component_y,
                        component_z=component_z,
                        **kwargs,
                    )

        return result


@dataclasses.dataclass
class Layer(kgpy.mixin.Copyable):
    material: typ.Optional[kgpy.labeled.Array] = None
    thickness: typ.Optional[kgpy.labeled.Array] = None
    num_periods: int = 1

    def __eq__(self: LayerT, other: LayerT):
        if not super().__eq__(other):
            return False
        if not isinstance(other, Layer):
            return False
        if not np.all(self.material == other.material):
            return False
        if not np.all(self.thickness == other.thickness):
            return False
        if not (self.num_periods == other.num_periods):
            return False
        return True

    def plot(
            self: LayerT,
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
            self: MultilayerMirrorT,
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
    transmissivity_function: typ.Optional[kgpy.function.Array] = None
    # efficiency_data: typ.Optional[u.Quantity] = None
    # wavelength_data: typ.Optional[u.Quantity] = None

    def transmissivity(self: MeasuredMultilayerMirrorT, ray: rays.RayVector) -> kgpy.uncertainty.ArrayLike:
        return self.transmissivity_function.interp_barycentric_linear(
            input_new=vectors.InputAngleVector(wavelength=ray.wavelength, angle_input_x=None, angle_input_y=None),
            axis=['wavelength'],
        ).output
        # interp = scipy.interpolate.interp1d(self.wavelength_data, self.efficiency_data, bounds_error=False)
        # return interp(rays.wavelength.to(self.wavelength_data.unit)) * self.efficiency_data.unit

    def __eq__(self: MeasuredMultilayerMirrorT, other: MeasuredMultilayerMirrorT) -> bool:
        if not super().__eq__(other):
            return False
        if not (self.transmissivity_function.input == other.transmissivity_function.input).all():
            return False
        if not (self.transmissivity_function.output == other.transmissivity_function.output).all():
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

    def transmissivity_aluminum(self, ray: rays.RayVector) -> u.Quantity:
        absorption = self.xrt_aluminum.get_absorption_coefficient(ray.energy.to(u.eV).array.value) / u.cm
        absorption = kgpy.labeled.Array(absorption, axes=ray.energy.axes)
        transmissivity = np.exp(-absorption * self.thickness / ray.direction.z)
        return transmissivity

    def transmissivity_aluminum_oxide(self, ray: rays.RayVector) -> u.Quantity:
        absorption = self.xrt_aluminum_oxide.get_absorption_coefficient(ray.energy.array.to(u.eV).value) / u.cm
        absorption = kgpy.labeled.Array(absorption, axes=ray.energy.axes)
        transmissivity = np.exp(-absorption * 2 * self.thickness_oxide / ray.direction.z)
        return transmissivity

    def transmissivity(self, ray: rays.RayVector) -> u.Quantity:
        mesh_ratio = self.mesh_ratio.to(u.dimensionless_unscaled)
        return mesh_ratio * self.transmissivity_aluminum(ray) * self.transmissivity_aluminum_oxide(ray)

    def index_refraction(self, ray: rays.RayVector) -> u.Quantity:
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

    def transmissivity(self, ray: rays.RayVector) -> u.Quantity:
        transmissivity_function = kgpy.function.Array(
            input=self.wavelength_data,
            output=self.quantum_efficiency_data,
        )
        return transmissivity_function.interp_linear(rays.wavelength)
        # qe_interp = scipy.interpolate.interp1d(self.wavelength_data, self.quantum_efficiency_data)
        # qe = qe_interp(rays.wavelength.to(self.wavelength_data.unit)) * self.quantum_efficiency_data.unit
        # return qe

    def index_refraction(self, ray: rays.RayVector) -> u.Quantity:
        return 1 * u.dimensionless_unscaled


@dataclasses.dataclass
class CCDStern2004(Material):

    data_path: typ.ClassVar[pathlib.Path] = pathlib.Path(__file__).parent / 'stern_2004_model_1.50e05.csv'

    @property
    def dataframe(self) -> pandas.DataFrame:
        return pandas.read_csv(self.data_path, header=None)

    @property
    def quantum_efficiency_data(self):
        return kgpy.labeled.Array(self.dataframe[1].to_numpy() * u.dimensionless_unscaled, axes=['wavelength'])

    @property
    def wavelength_data(self):
        return kgpy.labeled.Array(self.dataframe[0].to_numpy() * u.AA, axes=['wavelength'])

    def transmissivity(self, ray: rays.RayVector) -> kgpy.labeled.Array:
        transmissivity_function = kgpy.function.Array(
            input=self.wavelength_data,
            output=self.quantum_efficiency_data,
        )
        return transmissivity_function.interp_linear(ray.wavelength).output
        # qe_interp = scipy.interpolate.interp1d(self.wavelength_data, self.quantum_efficiency_data)
        # qe = qe_interp(rays.wavelength.to(self.wavelength_data.unit)) * self.quantum_efficiency_data.unit
        # return qe

    def index_refraction(self, ray: rays.RayVector) -> u.Quantity:
        return 1 * u.dimensionless_unscaled
