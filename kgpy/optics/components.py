import typing as typ
import abc
import dataclasses
import matplotlib.lines
import matplotlib.axes
import astropy.units as u
import pandas
import kgpy.mixin
import kgpy.format
import kgpy.uncertainty
import kgpy.vector
import kgpy.transforms
from .surfaces import Surface

__all__ = ['Component', 'PistonComponent', 'TranslationComponent', 'CylindricalComponent']

SurfaceT = typ.TypeVar('SurfaceT', bound=Surface)


@dataclasses.dataclass
class Component(
    kgpy.mixin.Plottable,
    kgpy.mixin.Named,
    abc.ABC,
    typ.Generic[SurfaceT],
):

    @property
    @abc.abstractmethod
    def transform(self) -> kgpy.transforms.TransformList:
        return kgpy.transforms.TransformList()

    @property
    def surface(self) -> SurfaceT:
        surface = Surface()
        surface.name = self.name
        surface.transform = self.transform
        surface.plot_kwargs = {**surface.plot_kwargs, **self.plot_kwargs}
        return surface

    def plot(
            self,
            ax: matplotlib.axes.Axes,
            component_x: str = 'x',
            component_y: str = 'y',
            component_z: str = 'z',
            transform_extra: typ.Optional[kgpy.transforms.TransformList] = None,
            to_global: bool = False,
            plot_annotations: bool = True,
            annotation_text_y: float = 1.05,
            **kwargs,
    ) -> typ.List[matplotlib.lines.Line2D]:

        kwargs = {**self.plot_kwargs, **kwargs}
        return self.surface.plot(
            ax=ax,
            component_x=component_x,
            component_y=component_y,
            component_z=component_z,
            transform_extra=transform_extra,
            to_global=to_global,
            plot_annotations=plot_annotations,
            **kwargs,
        )


@dataclasses.dataclass
class PistonComponent(Component[SurfaceT]):
    piston: kgpy.uncertainty.ArrayLike = 0 * u.mm

    @property
    def transform(self) -> kgpy.transforms.TransformList:
        return super().transform + kgpy.transforms.TransformList([
            kgpy.transforms.Translation(kgpy.vector.Cartesian3D(z=-self.piston))
        ])

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        dataframe['piston'] = [kgpy.format.quantity(self.piston)]
        return dataframe


@dataclasses.dataclass
class TranslationComponent(Component[SurfaceT]):
    translation: kgpy.vector.Cartesian3D = dataclasses.field(default_factory=lambda: kgpy.vector.Cartesian3D() * u.mm)

    @property
    def transform(self) -> kgpy.transforms.TransformList:
        return super().transform + kgpy.transforms.TransformList([kgpy.transforms.Translation(self.translation)])

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        dataframe['translation'] = [kgpy.format.quantity(self.translation.array)]
        return dataframe


@dataclasses.dataclass
class CylindricalComponent(TranslationComponent[SurfaceT]):
    translation_cylindrical: kgpy.vector.Cylindrical = dataclasses.field(default_factory=kgpy.vector.Cylindrical)
    # cylindrical_radius: u.Quantity = 0 * u.mm
    # cylindrical_radius_error: u.Quantity = 0 * u.mm
    # cylindrical_azimuth: u.Quantity = 0 * u.deg
    # cylindrical_azimuth_error: u.Quantity = 0 * u.deg

    @property
    def transform(self) -> kgpy.transforms.TransformList:
        return super().transform + kgpy.transforms.TransformList([
            kgpy.transforms.Translation(self.translation_cylindrical.cartesian),
        ])

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        dataframe['cylindrical radius'] = [kgpy.format.quantity(self.translation_cylindrical.radius)]
        dataframe['cylindrical azimuth'] = [kgpy.format.quantity(self.translation_cylindrical.azimuth)]
        dataframe['cylindrical $z$'] = [kgpy.format.quantity(self.translation_cylindrical.z)]
        return dataframe
