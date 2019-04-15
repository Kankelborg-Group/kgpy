
import astropy.units as u

from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.system.field import Item as Base

__all__ = []


class Item(Base):
    
    def __init__(self, x: u.Quantity, y: u.Quantity, zos_field: ZOSAPI.SystemData.IField,
                 zos_config: ZOSAPI.Editors.MCE.IMultiConfigEditor):
        
        self.zos_field = zos_field
        self.zos_config = zos_config
        
        Base.__init__(self, x, y)
        
    @property
    def x(self) -> u.Quantity:
        return self.zos_field.X * u.deg
    
    @x.setter
    def x(self, value: u.Quantity):

        if self._x_op is None:
            
            pass
            

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
