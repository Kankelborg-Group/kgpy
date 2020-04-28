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

    @property
    def tilt(self) -> Tilt:
        return super().tilt

    @tilt.setter
    def tilt(self, value: Tilt):
        if not isinstance(value, Tilt):
            value = Tilt.promote(value)
        super(__class__, self.__class__).tilt.__set__(self, value)

    @property
    def decenter(self) -> Decenter:
        return super().decenter

    @decenter.setter
    def decenter(self, value: Decenter):
        if not isinstance(value, Decenter):
            value = Decenter.promote(value)
        super(__class__, self.__class__).decenter.__set__(self, value)

    def _tilt_first_setter(self, value: int):
        self._composite._lde_row.SurfaceData.Order = value
