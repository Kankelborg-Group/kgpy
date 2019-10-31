
import typing as tp
import astropy.units as u

from kgpy import optics
from kgpy.optics.zemax import ZOSAPI

from . import Wavelength, WavelengthList, Field, FieldList, Surface, OperationList

__all__ = ['Configuration']


class Configuration(optics.system.Configuration):

    def __init__(self, surfaces: tp.List[Surface] = None):

        super().__init__(surfaces)

        self.mce_ops_list = OperationList()
        self.mce_ops_list.configuration = self

    @classmethod
    def conscript(cls, config: optics.system.Configuration):

        zmx_config = cls(config.name)

        zmx_config.entrance_pupil_radius = config.entrance_pupil_radius
        # zmx_config.wavelengths = WavelengthList.config.wavelengths
        zmx_config.fields = FieldList.conscript(config.fields)

        for surf in config:

            zmx_surf = Surface.conscript(surf)

            zmx_config.append(zmx_surf)

    @property
    def system(self) -> optics.zemax.System:
        return super().system

    @system.setter
    def system(self, value: optics.zemax.System):
        super().system = value

    @property
    def fields(self) -> FieldList:
        return super().fields

    @fields.setter
    def fields(self, value: FieldList):
        super().fields = value

    @property
    def wavelengths(self) -> WavelengthList:
        return super().wavelengths

    @wavelengths.setter
    def wavelengths(self, value: WavelengthList):
        super().wavelengths = value

    @property
    def entrance_pupil_radius(self) -> u.Quantity:

        return super().entrance_pupil_radius

    @entrance_pupil_radius.setter
    def entrance_pupil_radius(self, value: u.Quantity):

        super().entrance_pupil_radius = value

        try:

            self.system.zos.SystemData.Aperture.ApertureType = ZOSAPI.SystemData.ZemaxApertureType.EntrancePuilDiameter

            self.system.zos.SystemData.Aperture.ApertureValue = 2 * value.to(self.system.lens_units).value

        except AttributeError:
            pass

