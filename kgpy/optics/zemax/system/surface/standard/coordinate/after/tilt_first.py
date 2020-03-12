import dataclasses
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import configuration

from .. import before

__all__ = ['TiltFirst']


@dataclasses.dataclass
class InstanceVarBase(before.tilt.InstanceVarBase):

    _value_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.CAOR,
        ),
        init=None,
        repr=None,
    )


class TiltFirst(before.TiltFirst, InstanceVarBase):

    def _value_setter(self, value: int):
        self.tilt_decenter.surface.lde_row.TiltDecenterData.AfterSurfaceOrder = value
