import typing as typ
import abc
import dataclasses
import matplotlib.lines
import matplotlib.axes
import astropy.units as u
import pandas
from kgpy import mixin, format, vector, transform as tfrm
from .surface import Surface

__all__ = ['Component', 'PistonComponent', 'TranslationComponent', 'CylindricalComponent']

SurfaceT = typ.TypeVar('SurfaceT', bound=Surface)


@dataclasses.dataclass
class Component(
    mixin.Plottable,
    mixin.Named,
    abc.ABC,
    typ.Generic[SurfaceT],
):

    @property
    @abc.abstractmethod
    def transform(self) -> tfrm.rigid.TransformList:
        return tfrm.rigid.TransformList()

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
            components: typ.Tuple[str, str] = ('x', 'y'),
            component_z: typ.Optional[str] = None,
            plot_kwargs: typ.Optional[typ.Dict[str, typ.Any]] = None,
            # color: typ.Optional[str] = None,
            # linewidth: typ.Optional[float] = None,
            # linestyle: typ.Optional[str] = None,
            transform_extra: typ.Optional[tfrm.rigid.TransformList] = None,
            to_global: bool = False,
            plot_annotations: bool = True,
            annotation_text_y: float = 1.05,
    ) -> typ.List[matplotlib.lines.Line2D]:
        if plot_kwargs is not None:
            plot_kwargs = {**self.plot_kwargs, **plot_kwargs}
        else:
            plot_kwargs = self.plot_kwargs
        return self.surface.plot(
            ax=ax,
            components=components,
            component_z=component_z,
            plot_kwargs=plot_kwargs,
            transform_extra=transform_extra,
            to_global=to_global,
            plot_annotations=plot_annotations,
        )


@dataclasses.dataclass
class PistonComponent(Component[SurfaceT]):
    piston: u.Quantity = 0 * u.mm
    piston_error: u.Quantity = 0 * u.mm

    @property
    def transform(self) -> tfrm.rigid.TransformList:
        return super().transform + tfrm.rigid.TransformList([
            tfrm.rigid.Translate(z=-(self.piston + self.piston_error))
        ])

    def view(self) -> 'PistonComponent':
        other = super().view()  # type: PistonComponent
        other.piston = self.piston
        other.piston_error = self.piston_error
        return other

    def copy(self) -> 'PistonComponent':
        other = super().copy()      # type: PistonComponent
        other.piston = self.piston.copy()
        other.piston_error = self.piston_error.copy()
        return other

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        dataframe['piston'] = [format.quantity(self.piston)]
        dataframe['piston error'] = [format.quantity(self.piston)]
        return dataframe


@dataclasses.dataclass
class TranslationComponent(Component[SurfaceT]):
    translation: tfrm.rigid.Translate = dataclasses.field(default_factory=tfrm.rigid.Translate)

    @property
    def transform(self) -> tfrm.rigid.TransformList:
        return super().transform + tfrm.rigid.TransformList([self.translation])

    def view(self) -> 'TranslationComponent':
        other = super().view()      # type: TranslationComponent
        other.translation = self.translation
        return other

    def copy(self) -> 'TranslationComponent':
        other = super().copy()      # type: TranslationComponent
        other.translation = self.translation.copy()
        return other

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        dataframe['translation'] = [format.quantity(self.translation.value.quantity)]
        return dataframe


@dataclasses.dataclass
class CylindricalComponent(PistonComponent[SurfaceT]):
    cylindrical_radius: u.Quantity = 0 * u.mm
    cylindrical_azimuth: u.Quantity = 0 * u.deg

    @property
    def transform(self) -> tfrm.rigid.TransformList:
        return super().transform + tfrm.rigid.TransformList([
            tfrm.rigid.TiltZ(self.cylindrical_azimuth),
            tfrm.rigid.Translate(x=self.cylindrical_radius),
        ])

    def view(self) -> 'CylindricalComponent':
        other = super().view()  # type: CylindricalComponent
        other.cylindrical_radius = self.cylindrical_radius
        other.cylindrical_azimuth = self.cylindrical_azimuth
        return other

    def copy(self) -> 'CylindricalComponent':
        other = super().copy()  # type: CylindricalComponent
        other.cylindrical_radius = self.cylindrical_radius.copy()
        other.cylindrical_azimuth = self.cylindrical_azimuth.copy()
        return other

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        dataframe['cylindrical radius'] = [format.quantity(self.cylindrical_radius)]
        dataframe['cylindrical azimuth'] = [format.quantity(self.cylindrical_azimuth)]
        return dataframe
