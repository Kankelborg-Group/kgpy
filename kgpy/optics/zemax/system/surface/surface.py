import dataclasses
import abc
import typing as typ
import win32com.client
import numpy as np
import astropy.units as u
from kgpy.component import Component
from kgpy.optics.system import name
import kgpy.optics.system.surface
from ... import ZOSAPI
from .. import system, configuration

__all__ = ['Surface']


@dataclasses.dataclass
class OperandBase:

    _thickness_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.THIC
        ),
        init=False,
        repr=False,
    )
    _is_active_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.IGNR
        ),
        init=False,
        repr=False,
    )
    _is_visible_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.SDRW
        ),
        init=False,
        repr=False,
    )


@dataclasses.dataclass
class Surface(Component[system.System], kgpy.optics.system.Surface, OperandBase, abc.ABC):

    def _update(self) -> typ.NoReturn:
        super()._update()
        self._update_lde_row()
        self.name = self.name
        self.thickness = self.thickness
        self.is_active = self.is_active
        self.is_visible = self.is_visible

    @property
    @abc.abstractmethod
    def _lde_row_type(self) -> ZOSAPI.Editors.LDE.SurfaceType:
        pass

    @property
    @abc.abstractmethod
    def _lde_row_data(self) -> ZOSAPI.Editors.LDE.ISurface:
        return win32com.client.CastTo(self._lde_row.SurfaceData, ZOSAPI.Editors.LDE.ISurface.__name__)

    def _update_lde_row(self) -> typ.NoReturn:
        try:
            settings = self._lde_row.GetSurfaceTypeSettings(self._lde_row_type)
            self._lde_row.ChangeType(settings)
        except AttributeError:
            pass

    @property
    def name(self) -> 'name.Name':
        return self._name

    @name.setter
    def name(self, value: 'name.Name'):
        self._name = value
        try:
            self._lde_row.Comment = str(value)
        except AttributeError:
            pass

    def _thickness_setter(self, value: float):
        self._lde_row.Thickness = value

    @property
    def thickness(self) -> u.Quantity:
        return self._thickness

    @thickness.setter
    def thickness(self, value: u.Quantity):
        self._thickness = value
        self._set_with_lens_units(value, self._thickness_setter, self._thickness_op)

    def _is_active_setter(self, value: bool):
        self._lde_row.IsActive = not value

    @property
    def is_active(self) -> 'np.ndarray[bool]':
        return self._is_active

    @is_active.setter
    def is_active(self, value: 'np.ndarray[bool]'):
        self._is_active = value
        self._set(~value, self._is_active_setter, self._is_active_op)

    def _is_visible_setter(self, value: bool):
        self._lde_row.DrawData.DoNotDrawThisSurface = value

    @property
    def is_visible(self) -> 'np.ndarray[bool]':
        return self._is_visible

    @is_visible.setter
    def is_visible(self, value: 'np.ndarray[bool]'):
        self._is_visible = value
        self._set(~value, self._is_visible_setter, self._is_visible_op)

    @property
    def _lde_index(self) -> int:
        surfaces = list(self._composite.surfaces)
        return surfaces.index(self) + 1

    @property
    def _lde_row(self) -> ZOSAPI.Editors.LDE.ILDERow[ZOSAPI.Editors.LDE.ISurface]:
        return self._composite._lde.GetSurfaceAt(self._lde_index)

    @property
    def _lens_units(self) -> u.Unit:
        return self._composite._lens_units

    def _set(
            self,
            value: typ.Any,
            setter: typ.Callable[[typ.Any], None],
            operand: configuration.SurfaceOperand,
            unit: u.Unit = None,
    ) -> typ.NoReturn:
        operand.surface = self
        try:
            self._composite._set(value, setter, operand, unit)
        except AttributeError:
            pass

    def _set_with_lens_units(
            self,
            value: typ.Any,
            setter: typ.Callable[[typ.Any], None],
            operand: configuration.SurfaceOperand,
    ) -> typ.NoReturn:
        try:
            self._set(value, setter, operand, self._lens_units)
        except AttributeError:
            pass
