import dataclasses
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import configuration, surface
from .. import coordinate


__all__ = ['Tilt']


@dataclasses.dataclass
class Tilt(surface.coordinate.Tilt['coordinate.TiltDecenter']):

    _x_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM,
            param_2=3,
        ),
        init=False,
        repr=False,
    )
    _y_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM,
            param_2=4,
        ),
        init=False,
        repr=False,
    )
    _z_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_factory=lambda: ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM,
            param_2=5,
        ),
        init=False,
        repr=False,
    )

    def _x_setter(self, value: float):
        self.composite.composite.lde_row.SurfaceData.TiltAbout_X = value

    def _y_setter(self, value: float):
        self.composite.composite.lde_row.SurfaceData.TiltAbout_Y = value

    def _z_setter(self, value: float):
        self.composite.composite.lde_row.SurfaceData.TiltAbout_Z = value
