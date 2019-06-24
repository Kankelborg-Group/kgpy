
import typing as tp

from kgpy import optics
from kgpy.optics.zemax import ZOSAPI

from . import Field

__all__ = ['FieldList']


class FieldList(optics.system.configuration.FieldList):
    
    def __init__(self, fields: tp.List[Field] = None):

        super().__init__(fields)

        self.configuration = None       # type: tp.Optional[optics.zemax.system.Configuration]

    @classmethod
    def conscript(cls, configuration: tp.Optional[optics.zemax.system.Configuration]):


        
    def append(self, field: Field):

        if len(self.items) < 1:
            zos_field = self.zos_arr.GetField(1)

        else:
            zos_field = self.zos_arr.AddField(0.0, 0.0, 1.0)
            
        f = Field(field.x, field.y, zos_field, self.zos_sys)
        
        f.van = field.van
        f.vdx = field.vdx
        f.vdy = field.vdy
        f.vcx = field.vcx
        f.vcy = field.vcy
        
        ArrayBase.append(self, f)
