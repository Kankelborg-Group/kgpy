
import typing as tp
import astropy.units as u

from kgpy.optics.zemax import ZOSAPI

from . import OperationList

__all__ = ['Operation']


class Operation:

    def __init__(self):

        self.operation_list = None          # type: tp.Optional[OperationList]

    @property
    def mce_operand(self) -> ZOSAPI.Editors.MCE.IMCERow:
        return self.operation_list

    @property
    def index(self) -> int:
        return self.operation_list.index(self)

    @property
    def mce_operand_type(self):
        return self._mce_operand_type

    @mce_operand_type.setter
    def mce_operand_type(self, value):
        self._mce_operand_type = value

        try: