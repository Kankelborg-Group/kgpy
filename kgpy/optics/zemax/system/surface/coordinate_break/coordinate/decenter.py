import dataclasses
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import configuration, surface

__all__ = ['Decenter']


@dataclasses.dataclass
class Decenter(surface.coordinate.Decenter[surface.CoordinateBreak]):

    _x_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM,
            param_2=1,
        ),
        init=False,
        repr=False,
    )
    _y_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM,
            param_2=2,
        ),
        init=False,
        repr=False,
    )

    def _x_setter(self, value: float):
        p = self.parent.parent

    def _y_setter(self, value: float):
        self.parent.parent.lde_row.TiltDecenterData.BeforeSurfaceDecenterY = value
