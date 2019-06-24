
import enum
import typing as tp
import astropy.units as u

from kgpy import optics
from kgpy.optics.zemax import ZOSAPI

from . import FieldList

__all__ = ['Field']


class Field(optics.system.configuration.Field):

    class MultiConfigOps(enum.IntEnum):

        x = enum.auto()
        y = enum.auto()
        weight = enum.auto()

        vdx = enum.auto()
        vdy = enum.auto()
        vcx = enum.auto()
        vcy = enum.auto()
        van = enum.auto()

    def __init__(self):

        super().__init__()

        self.field_list = None      # type: tp.Optional[FieldList]
        self.zos_field = None       # type: tp.Optional[ZOSAPI.SystemData.IField]

    @classmethod
    def conscript(cls, field: optics.system.configuration.Field) -> 'Field':

        zmx_field = cls()
        
        zmx_field.x = field.x
        zmx_field.y = field.y
        zmx_field.weight = field.weight
        
        zmx_field.vdx = field.vdx
        zmx_field.vdy = field.vdy
        zmx_field.vcx = field.vcx
        zmx_field.vcy = field.vcy
        zmx_field.van = field.van
        
        return zmx_field

    @property
    def field_list(self) -> FieldList:
        return self._field_list

    @field_list.setter
    def field_list(self, value: FieldList):
        self._field_list = value

        try:
            mce = self.field_list.configuration.system.zos.MCE
        except AttributeError:
            return

        try:
            mce.GetOperandAt(self.field_list.configuration.mce_op_num['x'])
        
        try:

            self._x_op = self.field_list.configuration.system.zos.MCE.AddOperand()
            self._x_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.XFIE)

            self._y_op = self.field_list.configuration.system.zos.MCE.AddOperand()
            self._y_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.YFIE)

            self._vdx_op = self.field_list.configuration.system.zos.MCE.AddOperand()
            self._vdx_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.FVDX)

            self._vdy_op = self.field_list.configuration.system.zos.MCE.AddOperand()
            self._vdy_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.FVDY)

            self._vcx_op = self.field_list.configuration.system.zos.MCE.AddOperand()
            self._vcx_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.FVCX)

            self._vcy_op = self.field_list.configuration.system.zos.MCE.AddOperand()
            self._vcy_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.FVCY)

            self._van_op = self.field_list.configuration.system.zos.MCE.AddOperand()
            self._van_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.FVAN)

        except AttributeError:
            pass

    @property
    def zos_field(self) -> tp.Optional[ZOSAPI.SystemData.IField]:

        try:
            zos_fields = self.field_list.configuration.system.zos.SystemData.Fields

        except AttributeError as e:
            print(e)
            return None

        try:
            zos_field = zos_fields.GetField(self.index + 1)

        except Exception as e:
            print(e)
            zos_field = zos_fields.AddField(0.0, 0.0, 1.0)

        return zos_field

    @zos_field.setter
    def zos_field(self, value: ZOSAPI.SystemData.IField):

        self._zos_field = value

        try:

            self._x_op.Param1 = value.FieldNumber
            self._y_op.Param1 = value.FieldNumber
            self._vdx_op.Param1 = value.FieldNumber
            self._vdy_op.Param1 = value.FieldNumber
            self._vcx_op.Param1 = value.FieldNumber
            self._vcy_op.Param1 = value.FieldNumber
            self._van_op.Param1 = value.FieldNumber

        except AttributeError:
            pass
        
        self.update()
        
    def update(self):
        
        self.x = self.x
        self.y = self.y
        self.weight = self.weight
        
        self.vdx = self.vdx
        self.vdy = self.vdy
        self.vcx = self.vcx
        self.vcy = self.vcy
        self.van = self.van
                
    @property
    def x(self) -> u.Quantity:
        return super().x
    
    @x.setter
    def x(self, value: u.Quantity):
        super().x = value

        try:
            self.zos_field.X = value.to(u.deg).value
        except AttributeError:
            pass

    @property
    def y(self) -> u.Quantity:
        return super().y

    @y.setter
    def y(self, value: u.Quantity):
        super().y = value

        try:
            self.zos_field.Y = value.to(u.deg).value
        except AttributeError:
            pass

    @property
    def weight(self) -> u.Quantity:
        return super().weight

    @weight.setter
    def weight(self, value: u.Quantity):
        super().weight = value

        try:
            self.zos_field.Weight = float(value)
        except AttributeError:
            pass

    @property
    def vdx(self) -> u.Quantity:
        return super().vdx

    @vdx.setter
    def vdx(self, value: u.Quantity):
        super().vdx = value

        try:
            self.zos_field.VDX = float(value)
        except AttributeError:
            pass
        
    @property
    def vdy(self) -> u.Quantity:
        return super().vdy

    @vdy.setter
    def vdy(self, value: u.Quantity):
        super().vdy = value

        try:
            self.zos_field.VDY = float(value)
        except AttributeError:
            pass

    @property
    def vcx(self) -> u.Quantity:
        return super().vcx

    @vcx.setter
    def vcx(self, value: u.Quantity):
        super().vcx = value

        try:
            self.zos_field.VCX = float(value)
        except AttributeError:
            pass

    @property
    def vcy(self) -> u.Quantity:
        return super().vcy

    @vcy.setter
    def vcy(self, value: u.Quantity):
        super().vcy = value

        try:
            self.zos_field.VCY = float(value)
        except AttributeError:
            pass
        
    @property
    def van(self) -> u.Quantity:
        return super().van

    @van.setter
    def van(self, value: u.Quantity):
        super().van = value

        try:
            self.zos_field.VAN = value.to(u.deg).value
        except AttributeError:
            pass
        


