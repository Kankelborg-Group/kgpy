
import typing as tp
import astropy.units as u

from kgpy.optics.zemax import ZOSAPI

from . import OperationList

__all__ = ['Operation']


class Operation:

    def __init__(self):

        self.operation_list = None          # type: tp.Optional[OperationList]

        self.operand_type = None
        self.param1 = None
        self.param2 = None

    @property
    def operand(self) -> tp.Optional[ZOSAPI.Editors.MCE.IMCERow]:

        try:
            mce = self.operation_list.configuration.system.zos.MCE
        except AttributeError:
            return None

        try:
            return mce.GetOperandAt(self.index)
        except Exception:
            mce.AddOperand()
            return self.operand

    @property
    def index(self) -> tp.Optional[int]:
        try:
            return self.operation_list.index(self)
        except AttributeError:
            return None

    @property
    def operand_type(self) -> ZOSAPI.Editors.MCE.MultiConfigOperandType:
        return self._mce_operand_type

    @operand_type.setter
    def operand_type(self, value: ZOSAPI.Editors.MCE.MultiConfigOperandType):
        self._mce_operand_type = value

        try:
            self.operand.Type = value
        except AttributeError:
            pass
        
    @property
    def param1(self) -> u.Quantity:
        return self._mce_param1
    
    @param1.setter
    def param1(self, value: u.Quantity):
        self._mce_param1 = value

        try:
            self.operand.Param1 = value.to(self.operation_list.configuration.system.lens_units).value

        except AttributeError:
            pass

    @property
    def param2(self) -> u.Quantity:
        return self._mce_param2

    @param2.setter
    def param2(self, value: u.Quantity):
        self._mce_param2 = value

        try:
            self.operand.Param2 = value.to(self.operation_list.configuration.system.lens_units).value

        except AttributeError:
            pass

    def update(self):

        self.operand_type = self.operand_type
        self.param1 = self.param1
