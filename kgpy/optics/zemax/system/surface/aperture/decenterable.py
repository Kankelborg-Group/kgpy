import dataclasses
from kgpy.optics.zemax import ZOSAPI
from ... import configuration

__all__ = ['Operands']


@dataclasses.dataclass
class Operands:

    _x_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.APDX,
        ),
        init=False,
        repr=False,
    )
    _y_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.APDY,
        ),
        init=False,
        repr=False,
    )


