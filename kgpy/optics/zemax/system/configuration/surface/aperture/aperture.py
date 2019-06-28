
import abc
import typing as tp
import astropy.units as u

from kgpy import optics
from kgpy.optics.zemax import ZOSAPI


__all__ = ['Aperture']

# need to initialize settings when appended onto system


class Aperture(optics.system.configuration.surface.Aperture):

    def __init__(self):

        super().__init__()

        self.aperture_type = ZOSAPI.Editors.LDE.SurfaceApertureTypes.none

        self.settings = None


    @property
    def aperture_type(self) -> ZOSAPI.Editors.LDE.SurfaceApertureTypes:
        return self._aperture_type

    @aperture_type.setter
    def aperture_type(self, value: ZOSAPI.Editors.LDE.SurfaceApertureTypes):
        self._aperture_type = value

        try:
            self.surface.zos_surface.ApertureData.CurrentType = value

        except AttributeError:
            pass

    @property
    def settings(self) -> tp.Optional[ZOSAPI.Editors.LDE.ISurfaceApertureType]:

        return self._settings

        # return self.surface.zos_surface.ApertureData.CurrentTypeSettings

    @settings.setter
    def settings(self, value: tp.Optional[ZOSAPI.Editors.LDE.ISurfaceApertureType]):

        self._settings = value

        try:
            self.surface.zos_surface.ApertureData.ChangeApertureTypeSettings(value)

        except AttributeError:
            pass

    @property
    def surface(self) -> optics.zemax.system.configuration.Surface:
        return super().surface

    @surface.setter
    def surface(self, value: optics.zemax.system.configuration.Surface):
        super().surface = value

    @property
    def decenter_x(self) -> u.Quantity:

        return super().decenter_x

    @decenter_x.setter
    def decenter_x(self, value: u.Quantity):

        super().decenter_x = value

        s = self.settings
        s.ApertureXDecenter = value.to(self.surface.configuration.system.lens_units).value
        self.settings = s

    @property
    def decenter_y(self) -> u.Quantity:

        return super().decenter_y

    @decenter_y.setter
    def decenter_y(self, value: u.Quantity):

        super().decenter_y = value

        s = self.settings
        s.ApertureYDecenter = value.to(self.surface.configuration.system.lens_units).value
        self.settings = s

    @property
    def is_obscuration(self):

        return self._is_obscuration

    @is_obscuration.setter
    def is_obscuration(self, value: bool):

        self._is_obscuration = value

        s = self.settings
        s.IsObscuration = value
        self.settings = s

    def update(self):

        self.aperture_type = self.aperture_type
        self.settings = self.settings

        self.decenter_x = self.decenter_x
        self.decenter_y = self.decenter_y
        self.is_obscuration = self.is_obscuration












