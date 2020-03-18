import dataclasses
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import configuration, surface

__all__ = ['Decenter']


@dataclasses.dataclass
class Decenter(surface.coordinate.Decenter):

    _x_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.APDX,
        ),
        init=False,
        repr=False,
    )
    _y_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.APDY,
        ),
        init=False,
        repr=False,
    )
