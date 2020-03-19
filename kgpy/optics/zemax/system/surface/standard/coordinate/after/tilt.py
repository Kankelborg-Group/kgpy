import dataclasses

from kgpy.optics.zemax import ZOSAPI
from ..... import configuration, surface

__all__ = ['Tilt']


@dataclasses.dataclass
class Tilt(surface.coordinate.Tilt):

    _x_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.CATX,
        ),
        init=False,
        repr=False,
    )
    _y_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.CATY,
        ),
        init=False,
        repr=False,
    )
    _z_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.CATZ,
        ),
        init=False,
        repr=False,
    )

    def _x_setter(self, value: float):
        self.tilt_decenter.surface.lde_row.TiltDecenterData.AfterSurfaceTiltX = value

    def _y_setter(self, value: float):
        self.tilt_decenter.surface.lde_row.TiltDecenterData.AfterSurfaceTiltY = value

    def _z_setter(self, value: float):
        self.tilt_decenter.surface.lde_row.TiltDecenterData.AfterSurfaceTiltZ = value
