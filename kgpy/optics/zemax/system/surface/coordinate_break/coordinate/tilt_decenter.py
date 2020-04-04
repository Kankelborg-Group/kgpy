import dataclasses
from kgpy.optics.zemax import ZOSAPI
from .... import configuration
from ... import coordinate, coordinate_break
from . import Tilt, Decenter

__all__ = ['TiltDecenter']


@dataclasses.dataclass
class TiltDecenter(coordinate.TiltDecenter['coordinate_break.CoordinateBreak']):

    _tilt_first_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM,
            param_2=6
        ),
        init=False,
        repr=False,
    )

    tilt: Tilt = dataclasses.field(default_factory=lambda: Tilt())
    decenter: Decenter = dataclasses.field(default_factory=lambda: Decenter())

    def _tilt_first_setter(self, value: int):
        self.composite.lde_row.SurfaceData.Order = value
