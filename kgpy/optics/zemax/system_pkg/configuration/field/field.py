
import enum
import typing as tp
import astropy.units as u

from kgpy import optics
from kgpy.optics.zemax import ZOSAPI

from . import FieldList

__all__ = ['Field']


class Field(optics.system.configuration.Field):

    x_op_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.XFIE
    y_op_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.YFIE
    weight_op_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.FLWT     # todo: check this constant
    vdx_op_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.FVDX
    vdy_op_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.FVDY
    vcx_op_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.FVCX
    vcy_op_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.FVCY
    van_op_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.FVAN

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

        self.x_op = kgpy.optics.zemax.system.system_module.configuration.Operation()
        self.y_op = kgpy.optics.zemax.system.system_module.configuration.Operation()
        self.weight_op = kgpy.optics.zemax.system.system_module.configuration.Operation()
        self.vdx_op = kgpy.optics.zemax.system.system_module.configuration.Operation()
        self.vdy_op = kgpy.optics.zemax.system.system_module.configuration.Operation()
        self.vcx_op = kgpy.optics.zemax.system.system_module.configuration.Operation()
        self.vcy_op = kgpy.optics.zemax.system.system_module.configuration.Operation()
        self.van_op = kgpy.optics.zemax.system.system_module.configuration.Operation()

        self.x_op.operand_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.XFIE
        self.y_op.operand_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.YFIE
        self.weight_op = ZOSAPI.Editors.MCE.MultiConfigOperandType.FLWT     # todo: check this constant
        self.vdx_op.operand_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.FVDX
        self.vdy_op.operand_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.FVDY
        self.vcx_op.operand_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.FVCX
        self.vcy_op.operand_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.FVCY
        self.van_op.operand_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.FVAN



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
    def zos_field(self) -> tp.Optional[ZOSAPI.SystemData.IField]:

        try:
            zos_fields = self.field_list.configuration.system.zos.SystemData.Fields

        except AttributeError:
            return None

        try:
            zos_field = zos_fields.GetField(self.index + 1)

        except Exception as e:
            print(e)
            zos_fields.AddField(0.0, 0.0, 1.0)
            zos_field = self.zos_field

        return zos_field

    @property
    def index(self) -> tp.Optional[int]:
        try:
            return self.field_list.index(self)
        except AttributeError:
            return None

    @property
    def x(self) -> u.Quantity:
        return super().x
    
    @x.setter
    def x(self, value: u.Quantity):
        super().x = value

        try:
            self.x_op.param1 = self.index + 1
            self.x_op.update()
        except TypeError:
            return

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
            self.y_op.param1 = self.index + 1
            self.y_op.update()
        except TypeError:
            return

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
            self.weight_op.param1 = self.index + 1
            self.weight_op.update()
        except TypeError:
            return

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
            self.vdx_op.param1 = self.index + 1
            self.vdx_op.update()
        except TypeError:
            return

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
            self.vdy_op.param1 = self.index + 1
            self.vdy_op.update()
        except TypeError:
            return

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
            self.vcx_op.param1 = self.index + 1
            self.vcx_op.update()
        except TypeError:
            return

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
            self.vcy_op.param1 = self.index + 1
            self.vcy_op.update()
        except TypeError:
            return

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
            self.van_op.param1 = self.index + 1
            self.van_op.update()
        except TypeError:
            return

        try:
            self.zos_field.VAN = value.to(u.deg).value
        except AttributeError:
            pass

    def update(self):

        self.x = self.x
        self.y = self.y
        self.weight = self.weight

        self.vdx = self.vdx
        self.vdy = self.vdy
        self.vcx = self.vcx
        self.vcy = self.vcy
        self.van = self.van
        


