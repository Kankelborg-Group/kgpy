
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
    def conscript(cls, field_list: optics.system.configuration.FieldList):

        zmx_field_list = cls()

        for field in field_list:

            zmx_field = optics.zemax.system.configuration.Field.conscript(field)

            zmx_field_list.append(zmx_field)

    @property
    def configuration(self) -> optics.zemax.system.Configuration:
        return self._configuration

    @configuration.setter
    def configuration(self, value: optics.zemax.system.Configuration):
        self._configuration = value
        
        self.update()

    def update(self):

        for field in self:
            field.update()
        
    def append(self, field: Field):

        super().append(field)
        
        field.field_list = self

        field.zos_field = self.next_zos_field()

