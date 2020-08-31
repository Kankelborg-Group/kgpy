import typing as typ
import abc
import dataclasses
import astropy.units as u
import pandas
from kgpy import mixin, format, vector, transform
from . import Surface

__all__ = ['Component', 'PistonComponent', 'TranslationComponent', 'CylindricalComponent']

SurfaceT = typ.TypeVar('SurfaceT', bound=Surface)


@dataclasses.dataclass
class Component(
    mixin.Named,
    abc.ABC,
    typ.Generic[SurfaceT],
):

    @property
    @abc.abstractmethod
    def transform(self) -> transform.rigid.TransformList:
        return transform.rigid.TransformList()

    @property
    def surface(self) -> SurfaceT:
        surface = Surface()
        surface.name = self.name
        surface.transform = self.transform
        return surface


@dataclasses.dataclass
class PistonComponent(Component[SurfaceT]):
    piston: u.Quantity = 0 * u.mm

    @property
    def transform(self) -> transform.rigid.TransformList:
        return super().transform + transform.rigid.TransformList([
            transform.rigid.Translate.from_components(z=-self.piston)
        ])

    def copy(self) -> 'PistonComponent':
        other = super().copy()      # type: PistonComponent
        other.piston = self.piston.copy()
        return other


@dataclasses.dataclass
class TranslationComponent(Component[SurfaceT]):
    translation: transform.rigid.Translate = dataclasses.field(default_factory=transform.rigid.Translate)

    @property
    def transform(self) -> transform.rigid.TransformList:
        return super().transform + transform.rigid.TransformList([self.translation])

    def copy(self) -> 'TranslationComponent':
        other = super().copy()      # type: TranslationComponent
        other.translation = self.translation.copy()
        return other


@dataclasses.dataclass
class CylindricalComponent(PistonComponent[SurfaceT]):
    cylindrical_radius: u.Quantity = 0 * u.mm
    cylindrical_azimuth: u.Quantity = 0 * u.deg

    @property
    def transform(self) -> transform.rigid.TransformList:
        return super().transform + transform.rigid.TransformList([
            transform.rigid.TiltZ(self.cylindrical_azimuth),
            transform.rigid.Translate.from_components(x=self.cylindrical_radius),
        ])

    def copy(self) -> 'CylindricalComponent':
        other = super().copy()  # type: CylindricalComponent
        other.cylindrical_radius = self.cylindrical_radius.copy()
        other.cylindrical_azimuth = self.cylindrical_azimuth.copy()
        return other

    @property
    def dataframe(self) -> pandas.DataFrame:
        return super().dataframe.append(pandas.DataFrame.from_dict(
            data={
                'cylindrical radius': format.quantity(self.cylindrical_radius),
                'cylindrical azimuth': format.quantity(self.cylindrical_azimuth),
            },
        ))
