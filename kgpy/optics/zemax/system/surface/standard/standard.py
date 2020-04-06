import dataclasses
import typing as typ
from astropy import units as u
from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI
from ... import configuration
from .. import surface, aperture as aperture_, material as material_
from . import coordinate

__all__ = ['Standard']

MaterialT = typ.TypeVar('MaterialT', bound=material_.Material)
ApertureT = typ.TypeVar('ApertureT', bound=aperture_.Aperture)


@dataclasses.dataclass
class InstanceVarBase:
    _radius_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.CRVT
        ),
        init=None,
        repr=None,
    )
    _conic_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.CONN
        ),
        init=None,
        repr=None,
    )


@dataclasses.dataclass
class Base(typ.Generic[MaterialT, ApertureT]):
    material: MaterialT = dataclasses.field(default_factory=material_.NoMaterial(), init=False, repr=False)
    aperture: ApertureT = dataclasses.field(default_factory=aperture_.NoAperture(), init=False, repr=False)
    transform_before: coordinate.before.TiltDecenter = dataclasses.field(
        default_factory=coordinate.before.TiltDecenter(), init=False, repr=False
    )
    transform_after: coordinate.after.TiltDecenter = dataclasses.field(
        default_factory=coordinate.after.TiltDecenter(), init=False, repr=False
    )


class Standard(Base[MaterialT, ApertureT], system.surface.Standard, InstanceVarBase, surface.Surface,
               typ.Generic[MaterialT, ApertureT]):

    def _update(self) -> typ.NoReturn:
        super()._update()
        self.radius = self.radius
        self.conic = self.conic
        self.material = self.material
        self.aperture = self.aperture
        self.transform_before = self.transform_before
        self.transform_after = self.transform_after

    def _radius_setter(self, value: float):
        self.lde_row.Radius = value

    @property
    def radius(self) -> u.Quantity:
        return self._radius

    @radius.setter
    def radius(self, value: u.Quantity):
        self._radius = value
        try:
            self.set(value, self._radius_setter, self._radius_op, self.lens_units)
        except AttributeError:
            pass

    def _conic_setter(self, value: float):
        self.lde_row.Conic = value

    @property
    def conic(self) -> u.Quantity:
        return self._conic

    @conic.setter
    def conic(self, value: u.Quantity):
        self._conic = value
        try:
            self.set(value, self._conic_setter, self._conic_op)
        except AttributeError:
            pass

    @property
    def material(self) -> MaterialT:
        return self._material

    @material.setter
    def material(self, value: MaterialT):
        value._composite = self
        self._material = value

    @property
    def aperture(self) -> ApertureT:
        return self._aperture

    @aperture.setter
    def aperture(self, value: ApertureT):
        value._composite = self
        self._aperture = value

    @property
    def transform_before(self) -> coordinate.before.TiltDecenter:
        return self._transform_before

    @transform_before.setter
    def transform_before(self, value: coordinate.before.TiltDecenter):
        value._composite = self
        self._transform_before = value

    @property
    def transform_after(self) -> coordinate.after.TiltDecenter:
        return self._transform_after

    @transform_after.setter
    def transform_after(self, value: coordinate.after.TiltDecenter):
        value._composite = self
        self._transform_after = value
