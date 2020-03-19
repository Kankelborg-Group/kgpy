import dataclasses

from kgpy.optics.zemax import ZOSAPI
from ..... import configuration, surface

__all__ = ['TiltFirst']


@dataclasses.dataclass
class TiltFirst(surface.coordinate.TiltFirst):

    _value_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.CAOR,
        ),
        init=False,
        repr=False,
    )

    def _value_setter(self, value: int):
        self.tilt_decenter.surface.lde_row.TiltDecenterData.AfterSurfaceOrder = value
