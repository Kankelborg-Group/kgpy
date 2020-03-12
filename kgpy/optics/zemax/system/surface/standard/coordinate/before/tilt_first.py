import dataclasses
import typing as typ

from kgpy.optics.system import coordinate
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system import configuration
from .tilt_decenter import TiltDecenter

__all__ = []


@dataclasses.dataclass
class InstanceVarBase:

    _value_op: configuration.SurfaceOperand = dataclasses.field(
        default_factory=lambda: configuration.SurfaceOperand(
            op_type=ZOSAPI.Editors.MCE.MultiConfigOperandType.CBOR,
        ),
        init=None,
        repr=None,
    )


@dataclasses.dataclass
class Base(coordinate.TiltFirst, InstanceVarBase):

    tilt_decenter: typ.Optional[TiltDecenter] = None


class TiltFirst(Base):

    def _update(self):
        self.value = self.value

    def _value_setter(self, value: int):
        self.tilt_decenter.surface.lde_row.TiltDecenterData.BeforeSurfaceOrder = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val
        try:
            self._value_op.surface_index = self.tilt_decenter.surface.lde_index
            self.tilt_decenter.surface.lde.system.set(int(val), self._value_setter, self._value_op)
        except AttributeError:
            pass

    @property
    def tilt_decenter(self) -> TiltDecenter:
        return self._tilt_decenter

    @tilt_decenter.setter
    def tilt_decenter(self, value: TiltDecenter):
        self._tilt_decenter = value
        self._update()
