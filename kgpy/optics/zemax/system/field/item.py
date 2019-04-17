
import astropy.units as u

from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.system.field import Item as Base

__all__ = []


class Item(Base):

    def __init__(self, x: u.Quantity, y: u.Quantity, zos_field: ZOSAPI.SystemData.IField,
                 zos_sys: ZOSAPI.IOpticalSystem):

        self.zos_field = zos_field
        self.zos_mce = zos_sys.MCE

        self._prep_mce()
        
        Base.__init__(self, x, y)
                
    @property
    def x(self) -> u.Quantity:
        return self.zos_field.X * u.deg
    
    @x.setter
    def x(self, value: u.Quantity):
        self.zos_field.X = value.to(u.deg).value

    @property
    def y(self) -> u.Quantity:
        return self.zos_field.Y * u.deg

    @y.setter
    def y(self, value: u.Quantity):
        self.zos_field.Y = value.to(u.deg).value

    @property
    def vdx(self) -> float:
        return self.zos_field.VDX

    @vdx.setter
    def vdx(self, value: float):
        self.zos_field.VDX = value
        
    @property
    def vdy(self) -> float:
        return self.zos_field.VDY

    @vdy.setter
    def vdy(self, value: float):
        self.zos_field.VDY = value

    @property
    def vcx(self) -> float:
        return self.zos_field.VCX

    @vcx.setter
    def vcx(self, value: float):
        self.zos_field.VCX = value

    @property
    def vcy(self) -> float:
        return self.zos_field.VCY

    @vcy.setter
    def vcy(self, value: float):
        self.zos_field.VCY = value
        
    @property
    def van(self) -> u.Quantity:
        return self.zos_field.VAN * u.deg

    @van.setter
    def van(self, value: u.Quantity):
        self.zos_field.VAN = value.to(u.deg).value

    def _prep_mce(self):
        
        self._x_op = self.zos_mce.AddOperand()
        self._x_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.XFIE)
        self._x_op.Param1 = self.zos_field.FieldNumber

        self._y_op = self.zos_mce.AddOperand()
        self._y_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.YFIE)
        self._y_op.Param1 = self.zos_field.FieldNumber

        self._vdx_op = self.zos_mce.AddOperand()
        self._vdx_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.FVDX)
        self._vdx_op.Param1 = self.zos_field.FieldNumber

        self._vdy_op = self.zos_mce.AddOperand()
        self._vdy_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.FVDY)
        self._vdy_op.Param1 = self.zos_field.FieldNumber

        self._vcx_op = self.zos_mce.AddOperand()
        self._vcx_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.FVCX)
        self._vcx_op.Param1 = self.zos_field.FieldNumber

        self._vcy_op = self.zos_mce.AddOperand()
        self._vcy_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.FVCY)
        self._vcy_op.Param1 = self.zos_field.FieldNumber

        self._van_op = self.zos_mce.AddOperand()
        self._van_op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.FVAN)
        self._van_op.Param1 = self.zos_field.FieldNumber
