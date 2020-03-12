import dataclasses
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import configuration

from .. import before

__all__ = ['Tilt']


@dataclasses.dataclass
class InstanceVarBase(before.tilt.InstanceVarBase):

    _x_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.CATX,
        ),
        init=None,
        repr=None,
    )
    _y_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.CATY,
        ),
        init=None,
        repr=None,
    )
    _z_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.CATZ,
        ),
        init=None,
        repr=None,
    )


class Tilt(before.Tilt, InstanceVarBase):

    def _x_setter(self, value: float):
        self.tilt_decenter.surface.lde_row.TiltDecenterData.AfterSurfaceTiltX = value

    def _y_setter(self, value: float):
        self.tilt_decenter.surface.lde_row.TiltDecenterData.AfterSurfaceTiltY = value

    def _z_setter(self, value: float):
        self.tilt_decenter.surface.lde_row.TiltDecenterData.AfterSurfaceTiltZ = value
