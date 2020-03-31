import dataclasses
import typing as typ
import numpy as np

from ... import ZOSAPI
from .. import Child
from .editor import Editor

__all__ = ['Operand']


@dataclasses.dataclass
class Base:

    op_factory: typ.Optional[typ.Callable[[], 'ZOSAPI.Editors.MCE.MultiConfigOperandType']] = None
    data: np.ndarray = None
    param_1: int = 0
    param_2: int = 0
    param_3: int = 0


class Operand(Child[Editor], Base):

    def _update(self) -> None:
        self.op_factory = self.op_factory
        self.data = self.data
        self.param_1 = self.param_1
        self.param_2 = self.param_2
        self.param_3 = self.param_3

    @property
    def op_factory(self) -> typ.Callable[[], ZOSAPI.Editors.MCE.MultiConfigOperandType]:
        return self._op_factory

    @op_factory.setter
    def op_factory(self, value: typ.Callable[[], ZOSAPI.Editors.MCE.MultiConfigOperandType]):
        self._op_factory = value
        try:
            self.mce_row.ChangeType(value())
        except AttributeError:
            pass

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        self._data = value
        try:
            value = np.broadcast_to(value, self.parent.parent.config_broadcast.shape).flat
            for i, v in enumerate(value):
                cell = self.mce_row.GetOperandCell(i + 1)
                v = v.item()
                if isinstance(v, float):
                    cell.Value = v
                elif isinstance(v, int):
                    cell.IntegerValue = v
                elif isinstance(v, bool):
                    cell.IntegerValue = int(v)
                else:
                    raise NotImplementedError

        except AttributeError:
            pass

    @property
    def param_1(self) -> int:
        return self._param_1

    @param_1.setter
    def param_1(self, value: int):
        self._param_1 = value
        try:
            self.mce_row.Param1 = value
        except AttributeError:
            pass

    @property
    def param_2(self) -> int:
        return self._param_2

    @param_2.setter
    def param_2(self, value: int):
        self._param_2 = value
        try:
            self.mce_row.Param2 = value
        except AttributeError:
            pass

    @property
    def param_3(self) -> int:
        return self._param_3

    @param_3.setter
    def param_3(self, value: int):
        self._param_3 = value
        try:
            self.mce_row.Param3 = value
        except AttributeError:
            pass

    @property
    def mce_index(self) -> int:
        return self.parent.index(self)

    @property
    def mce_row(self) -> ZOSAPI.Editors.MCE.IMCERow:
        return self.parent.parent.zemax_system.MCE.GetOperandAt(self.mce_index)


