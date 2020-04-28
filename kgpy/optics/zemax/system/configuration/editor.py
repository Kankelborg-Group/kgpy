import dataclasses
import typing as typ
from kgpy.component import Component
from ... import ZOSAPI, system
from . import operand

__all__ = ['Editor']


@dataclasses.dataclass
class Base:

    _operands: typ.List['operand.Operand'] = dataclasses.field(default_factory=lambda: [])


class Editor(Component['system.System'], Base):

    def _update(self) -> typ.NoReturn:
        self._operands = self._operands

    @property
    def _operands(self) -> typ.List['operand.Operand']:
        return self.__operands

    @_operands.setter
    def _operands(self, value: typ.List['operand.Operand']):
        for v in value:
            v._composite = self
        self.__operands = value

        try:
            while self._zemax_mce.NumberOfOperands != len(self._operands):
                if self._zemax_mce.NumberOfOperands < len(self._operands):
                    self._zemax_mce.AddOperand()
                else:
                    self._zemax_mce.RemoveOperandAt(self._zemax_mce.NumberOfOperands)
        except AttributeError:
            pass


    @property
    def _zemax_mce(self) -> ZOSAPI.Editors.MCE.IMultiConfigEditor:
        return self._composite._zemax_system.MCE

    def index(self, op: 'operand.Operand') -> int:
        return self._operands.index(op)

    def insert(self, index: int, value: 'operand.Operand') -> typ.NoReturn:
        value._composite = self
        self._operands.insert(index, value)
        self._zemax_mce.InsertNewOperandAt(index)

    def append(self, value: 'operand.Operand') -> typ.NoReturn:
        value._composite = self
        self._operands.append(value)
        self._zemax_mce.AddOperand()

    def pop(self, index: int) -> 'operand.Operand':
        value = self._operands.pop(index)
        value._composite = None
        self._zemax_mce.RemoveOperandAt(index)
        return value

    def __getitem__(self, item: typ.Union[int, slice]) -> typ.Union['operand.Operand', typ.Iterable['operand.Operand']]:
        return self._operands.__getitem__(item)

    def __setitem__(self, key: typ.Union[int, slice], value: typ.Union['operand.Operand', typ.Iterable['operand.Operand']]) -> None:
        if isinstance(value, operand.Operand):
            value._composite = self
        else:
            for v in value:
                v._composite = self
        self._operands.__setitem__(key, value)

    def __iter__(self):
        return self._operands.__iter__()



