import dataclasses
import typing as typ

from ... import ZOSAPI
from .. import system
from . import operand

__all__ = ['Editor']


@dataclasses.dataclass
class Base:

    _operands: typ.List['operand.Operand'] = dataclasses.field(default_factory=lambda: [])
    system: typ.Optional['system.System'] = None


class Editor(Base):

    def _update(self) -> None:
        self._operands = self._operands

    @property
    def _operands(self) -> typ.List['operand.Operand']:
        return self.__operands

    @_operands.setter
    def _operands(self, value: typ.List['operand.Operand']):
        self.__operands = value

        while self._zemax_mce.NumberOfOperands != len(self._operands):
            if self._zemax_mce.NumberOfOperands < len(self._operands):
                self._zemax_mce.AddOperand()
            else:
                self._zemax_mce.RemoveOperandAt(self._zemax_mce.NumberOfOperands)
        for v in value:
            v.mce = self

    @property
    def system(self) -> 'system.System':
        return self._system

    @system.setter
    def system(self, value: 'system.System'):
        self._system = value
        self._update()

    @property
    def _zemax_mce(self) -> ZOSAPI.Editors.MCE.IMultiConfigEditor:
        return self.system.zemax_system.MCE

    def index(self, op: 'operand.Operand') -> int:
        return self._operands.index(op)

    def insert(self, index: int, value: 'operand.Operand') -> None:
        self._operands.insert(index, value)
        self._zemax_mce.InsertNewOperandAt(index)
        value.mce = self

    def append(self, value: 'operand.Operand') -> None:
        self._operands.append(value)
        self._zemax_mce.AddOperand()
        value.mce = self

    def pop(self, index: int) -> 'operand.Operand':
        value = self._operands.pop(index)
        self._zemax_mce.RemoveOperandAt(index)
        value.mce = None
        return value

    def __getitem__(self, item: typ.Union[int, slice]) -> typ.Union['operand.Operand', typ.Iterable['operand.Operand']]:
        return self._operands.__getitem__(item)

    def __setitem__(self, key: typ.Union[int, slice], value: typ.Union['operand.Operand', typ.Iterable['operand.Operand']]) -> None:
        self._operands.__setitem__(key, value)
        if isinstance(value, operand.Operand):
            value.mce = self
        else:
            for v in value:
                v.mce = self

    def __iter__(self):
        return self._operands.__iter__()



