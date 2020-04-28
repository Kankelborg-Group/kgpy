import dataclasses
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import configuration, surface
from .. import coordinate

__all__ = ['Decenter']


@dataclasses.dataclass
class Decenter(surface.coordinate.Decenter['coordinate.TiltDecenter']):

    _x_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM,
            param_2=1,
        ),
        init=False,
        repr=False,
    )
    _y_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM,
            param_2=2,
        ),
        init=False,
        repr=False,
    )

    def _x_setter(self, value: float):
        self._composite._composite._lde_row.SurfaceData.Decenter_X = value

    def _y_setter(self, value: float):
        self._composite._composite._lde_row.SurfaceData.Decenter_Y = value
