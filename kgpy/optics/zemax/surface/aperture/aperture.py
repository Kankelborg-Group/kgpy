
from abc import ABC, abstractmethod
import astropy.units as u

from kgpy import optics
from kgpy.optics.zemax.ZOSAPI.Editors.LDE import SurfaceApertureTypes, ISurfaceApertureType, \
    ISurfaceApertureCircular

__all__ = ['Aperture']


class Aperture(ABC):

    def __init__(self, surf: 'optics.ZmxSurface'):

        self.surf = surf

        self.is_obscuration = False

        self.settings = self.surf.main_row.ApertureData.CreateApertureTypeSettings(self.aperture_type)

    @property
    @abstractmethod
    def aperture_type(self) -> SurfaceApertureTypes:
        pass

    @property
    @abstractmethod
    def settings(self) -> ISurfaceApertureType:

        return self.surf.main_row.ApertureData.CurrentTypeSettings

    @settings.setter
    @abstractmethod
    def settings(self, val: ISurfaceApertureType) -> None:

        self.surf.main_row.ApertureData.ChangeApertureTypeSettings(val)

    @property
    def decenter_x(self) -> u.Quantity:

        s = self.settings  # type: ISurfaceApertureCircular

        return s.ApertureXDecenter * self.surf.sys.lens_units

    @decenter_x.setter
    def decenter_x(self, val: u.Quantity):

        s = self.settings
        s.ApertureXDecenter = val.to(self.surf.sys.lens_units).value
        self.settings = s

    @property
    def decenter_y(self) -> u.Quantity:

        s = self.settings       # type: ISurfaceApertureCircular

        return s.ApertureYDecenter * self.surf.sys.lens_units

    @decenter_y.setter
    def decenter_y(self, val: u.Quantity):

        s = self.settings
        s.ApertureYDecenter = val.to(self.surf.sys.lens_units).value
        self.settings = s








