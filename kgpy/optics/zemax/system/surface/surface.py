import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
from kgpy.component import Component
import kgpy.optics.system.surface
from ... import ZOSAPI
from .. import util, configuration
from . import name, coordinate, editor

__all__ = ['Surface', 'add_surfaces_to_zemax_system']


@dataclasses.dataclass
class OperandBase:

    _is_active_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.IGNR
        ),
        init=None,
        repr=None
    )

    _is_visible_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.SDRW
        ),
        init=None,
        repr=None
    )


class Surface(Component[editor.Editor], kgpy.optics.system.system.Surface, OperandBase):

    def _update(self) -> typ.NoReturn:
        super()._update()
        self.name = self.name
        self.thickness = self.thickness
        self.is_stop = self._is_stop
        self.is_active = self.is_active
        self.is_visible = self.is_visible

    @property
    def _transform(self) -> coordinate.Transform:
        return self.__transform

    @_transform.setter
    def _transform(self, value: coordinate.Transform):
        self.__transform = value
        value.surface = self

    @property
    def name(self) -> 'name.Name':
        return self._name

    @name.setter
    def name(self, value: 'name.Name'):
        self._name = value
        value.surface = self

    @property
    def is_stop(self) -> bool:
        return self._is_stop

    @is_stop.setter
    def is_stop(self, value: bool):
        self._is_stop = value
        try:
            self.lde_row.IsStop = value
        except AttributeError:
            pass

    def _is_active_setter(self, value: bool):
        self.lde_row.IsActive = not value

    @property
    def is_active(self) -> 'np.ndarray[bool]':
        return self._is_active

    @is_active.setter
    def is_active(self, value: 'np.ndarray[bool]'):
        self._is_active = value
        try:
            self._is_active_op.surface_index = self.lde_index
            self.composite.composite.set(np.logical_not(value), self._is_active_setter, self._is_active_op)
        except AttributeError:
            pass

    def _is_visible_setter(self, value: bool):
        self.lde_row.DrawData.DoNotDrawThisSurface = value

    @property
    def is_visible(self) -> 'np.ndarray[bool]':
        return self._is_visible

    @is_visible.setter
    def is_visible(self, value: 'np.ndarray[bool]'):
        self._is_visible = value
        try:
            self._is_visible_op.surface_index = self.lde_index
            self.composite.composite.set(np.logical_not(value), self._is_visible_setter, self._is_visible_op)
        except AttributeError:
            pass

    @property
    def lde_index(self) -> int:
        return self.composite.index(self) + 1

    @property
    def lde_row(self) -> ZOSAPI.Editors.LDE.ILDERow[ZOSAPI.Editors.LDE.ISurface]:
        return self.composite.composite.zemax_system.LDE.GetSurfaceAt(self.lde_index)

    @property
    def lens_units(self) -> u.Unit:
        return self.composite.composite.lens_units

    def set(
            self,
            value: typ.Any,
            setter: typ.Callable[[typ.Any], None],
            operand: configuration.SurfaceOperand,
            unit: u.Unit = None,
    ) -> typ.NoReturn:
        operand.surface = self
        self.composite.composite.set(value, setter, operand, unit)



def add_surfaces_to_zemax_system(
        zemax_system: ZOSAPI.IOpticalSystem,
        surfaces: 'typ.Iterable[surface.Surface]',
        configuration_shape: typ.Tuple[int],
        zemax_units: u.Unit,

):

    op_comment = ZOSAPI.Editors.MCE.MultiConfigOperandType.MCOM
    op_thickness = ZOSAPI.Editors.MCE.MultiConfigOperandType.THIC
    op_is_visible = ZOSAPI.Editors.MCE.MultiConfigOperandType.SDRW

    unit_thickness = zemax_units

    surfaces = list(surfaces)
    num_surfaces = len(surfaces)
    while zemax_system.LDE.NumberOfSurfaces < num_surfaces + 1:
        zemax_system.LDE.AddSurface()
    
    for s in range(num_surfaces):
        
        surface_index = s + 1
        surf = surfaces[s]
        
        util.set_str(zemax_system, surf.name.__str__(), configuration_shape, op_comment, surface_index)
        util.set_float(zemax_system, surf.thickness, configuration_shape, op_thickness, unit_thickness,
                       surface_index)
        util.set_int(zemax_system, not surf.is_visible, configuration_shape, op_is_visible, surface_index)
        
        if isinstance(surf, surface.Standard):
            standard.add_to_zemax_system(zemax_system, surf, surface_index, configuration_shape, zemax_units)

        elif isinstance(surf, surface.CoordinateBreak):
            coordinate_break.add_to_zemax_system(zemax_system, surf, surface_index, configuration_shape, zemax_units)
