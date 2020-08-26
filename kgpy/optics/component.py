import typing as typ
import abc
import dataclasses
import astropy.units as u
import pandas
from kgpy import mixin, format, transform, optics

__all__ = ['Component', 'DummyComponent', 'StandardComponent', 'RelativeComponent', 'CylindricalComponent']

SurfaceT = typ.TypeVar('SurfaceT', bound=optics.Surface)
StandardT = typ.TypeVar('StandardT', bound=optics.surface.Standard)


@dataclasses.dataclass
class Component(
    mixin.Named,
    abc.ABC,
    typ.Generic[SurfaceT],
):
    @property
    @abc.abstractmethod
    def _surface_type(self) -> typ.Type[SurfaceT]:
        pass

    @property
    def surface(self) -> SurfaceT:
        surface = self._surface_type()
        surface.name = self.name
        return surface


@dataclasses.dataclass
class DummyComponent(Component):
    thickness: u.Quantity = 100 * u.mm

    @property
    def _surface_type(self) -> typ.Type[optics.surface.CoordinateTransform]:
        return optics.surface.CoordinateTransform

    @property
    def surface(self) -> optics.surface.CoordinateTransform:
        surface = super().surface
        surface.thickness = self.thickness
        return surface

    def copy(self) -> 'DummyComponent':
        other = super().copy()  # type: DummyComponent
        other.thickness = self.thickness.copy()
        return other


@dataclasses.dataclass
class StandardComponent(Component[StandardT]):

    @property
    def _surface_type(self) -> typ.Type[StandardT]:
        return optics.surface.Standard


@dataclasses.dataclass
class RelativeComponent(StandardComponent[StandardT]):

    @property
    def transform(self) -> transform.rigid.TransformList:
        return transform.rigid.TransformList()

    @property
    def surface(self) -> StandardT:
        surface = super().surface   # type: StandardT
        surface.transform_before = self.transform
        surface.transform_after = self.transform.inverse
        return surface


@dataclasses.dataclass
class PistonComponent(RelativeComponent[StandardT]):
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
class CylindricalComponent(PistonComponent[StandardT]):
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
