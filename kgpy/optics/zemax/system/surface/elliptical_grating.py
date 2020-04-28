import dataclasses
import typing as typ
from astropy import units as u
from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI
from .. import configuration
from . import DiffractionGrating

__all__ = ['EllipticalGrating1']


@dataclasses.dataclass
class InstanceVarBase:
    _a_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM,
            param_2=3,
        ),
        init=None,
        repr=None,
    )
    _b_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM,
            param_2=4,
        ),
        init=None,
        repr=None,
    )
    _c_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM,
            param_2=5,
        ),
        init=None,
        repr=None,
    )
    _alpha_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM,
            param_2=6,
        ),
        init=None,
        repr=None,
    )
    _beta_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM,
            param_2=7,
        ),
        init=None,
        repr=None,
    )

    _a_unit = u.dimensionless_unscaled
    _b_unit = u.dimensionless_unscaled
    _alpha_unit = u.dimensionless_unscaled
    _beta_unit = u.dimensionless_unscaled


@dataclasses.dataclass
class EllipticalGrating1(system.surface.EllipticalGrating1, InstanceVarBase, DiffractionGrating, ):

    def _update(self) -> typ.NoReturn:
        super()._update()
        self.a = self.a
        self.b = self.b
        self.c = self.c
        self.alpha = self.alpha
        self.beta = self.beta

    def _get_type(self) -> ZOSAPI.Editors.LDE.SurfaceType:
        return ZOSAPI.Editors.LDE.SurfaceType.EllipticalGrating1

    @property
    def _lde_row(self) -> ZOSAPI.Editors.LDE.ILDERow[ZOSAPI.Editors.LDE.ISurfaceEllipticalGrating1]:
        return super()._lde_row

    def _a_setter(self, value: float):
        self._lde_row.SurfaceData.A = value

    @property
    def a(self) -> u.Quantity:
        return self._a

    @a.setter
    def a(self, value: u.Quantity):
        self._a = value
        self._set(value, self._a_setter, self._a_op, self._a_unit)

    def _b_setter(self, value: float):
        self._lde_row.SurfaceData.B = value

    @property
    def b(self) -> u.Quantity:
        return self._b

    @b.setter
    def b(self, value: u.Quantity):
        self._b = value
        self._set(value, self._b_setter, self._b_op, self._b_unit)

    def _c_setter(self, value: float):
        self._lde_row.SurfaceData.C = value

    @property
    def c(self) -> u.Quantity:
        return self._c

    @c.setter
    def c(self, value: u.Quantity):
        self._c = value
        self._set_with_lens_units(value, self._c_setter, self._c_op)

    def _alpha_setter(self, value: float):
        self._lde_row.SurfaceData.Alpha = value

    @property
    def alpha(self) -> u.Quantity:
        return self._alpha

    @alpha.setter
    def alpha(self, value: u.Quantity):
        self._alpha = value
        self._set(value, self._alpha_setter, self._alpha_op, self._alpha_unit)

    def _beta_setter(self, value: float):
        self._lde_row.SurfaceData.Beta = value

    @property
    def beta(self) -> u.Quantity:
        return self._beta

    @beta.setter
    def beta(self, value: u.Quantity):
        self._beta = value
        self._set(value, self._beta_setter, self._beta_op, self._beta_unit)
