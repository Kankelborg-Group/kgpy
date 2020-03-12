import dataclasses

from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import configuration
from .. import before

__all__ = ['Decenter']


@dataclasses.dataclass
class InstanceVarBase(before.decenter.InstanceVarBase):
    _x_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.CADX,
        ),
        init=None,
        repr=None,
    )
    _y_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.CADY,
        ),
        init=None,
        repr=None,
    )


class Decenter(before.decenter.Decenter, InstanceVarBase):

    def _x_setter(self, value: float):
        self.tilt_decenter.surface.lde_row.TiltDecenterData.AfterSurfaceDecenterX = value

    def _y_setter(self, value: float):
        self.tilt_decenter.surface.lde_row.TiltDecenterData.AfterSurfaceDecenterX = value
