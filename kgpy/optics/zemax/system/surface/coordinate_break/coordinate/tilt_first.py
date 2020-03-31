import dataclasses

from kgpy.optics.zemax import ZOSAPI
from ..... import configuration
from .... import coordinate
from .. import before

__all__ = ['TiltFirst']


@dataclasses.dataclass
class TiltFirst(coordinate.TiltFirst['before.tilt_decenter.TiltDecenter[surface.Standard]']):

    _value_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM,
            param_2=6
        ),
        init=False,
        repr=False,
    )

    def _value_setter(self, value: int):
        self.parent.lde_row.SurfaceData.Order = value
