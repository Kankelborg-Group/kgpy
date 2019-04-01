
from abc import ABC, abstractmethod
import astropy.units as u

from kgpy.math import Vector
from kgpy import optics
from kgpy.optics.zemax.ZOSAPI.Editors.LDE import SurfaceApertureTypes, ISurfaceApertureType, \
    ISurfaceApertureCircular, ISurfaceApertureRectangular, ISurfaceApertureSpider, ISurfaceApertureUser

__all__ = ['ZmxAperture', 'Circular', 'Rectangular', 'Spider']


class ZmxAperture(ABC):

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


class Rectangular(ZmxAperture, optics.Rectangular):

    def __init__(self, half_width_x: u.Quantity, half_width_y: u.Quantity, surf: 'optics.ZmxSurface'):

        ZmxAperture.__init__(self, surf)

        optics.Rectangular.__init__(self, half_width_x, half_width_y)

    @property
    def aperture_type(self) -> SurfaceApertureTypes:

        if self.is_obscuration:
            return SurfaceApertureTypes.RectangularObscuration

        else:
            return SurfaceApertureTypes.RectangularAperture

    @property
    def settings(self) -> ISurfaceApertureRectangular:

        s = self.surf.main_row.ApertureData.CurrentTypeSettings

        if self.is_obscuration:
            return s._S_RectangularObscuration

        else:
            return s._S_RectangularAperture

    @settings.setter
    def settings(self, val: ISurfaceApertureRectangular) -> None:
        ZmxAperture.settings.fset(val)

    @property
    def half_width_x(self) -> u.Quantity:

        return self.settings.XHalfWidth * self.surf.sys.lens_units

    @half_width_x.setter
    def half_width_x(self, val: u.Quantity):

        s = self.settings
        s.XHalfWidth = val.to(self.surf.sys.lens_units).value
        self.settings = s

    @property
    def half_width_y(self) -> u.Quantity:

        return self.settings.YHalfWidth * self.surf.sys.lens_units

    @half_width_y.setter
    def half_width_y(self, val: u.Quantity):

        s = self.settings
        s.YHalfWidth = val.to(self.surf.sys.lens_units).value
        self.settings = s


class Circular(ZmxAperture, optics.Circular):

    def __init__(self, min_radius: u.Quantity, max_radius: u.Quantity, surf: 'optics.ZmxSurface'):

        ZmxAperture.__init__(self, surf)

        optics.Circular.__init__(self, min_radius, max_radius)

    @property
    def aperture_type(self) -> SurfaceApertureTypes:

        if self.is_obscuration:
            return SurfaceApertureTypes.CircularObscuration

        else:
            return SurfaceApertureTypes.CircularAperture

    @property
    def settings(self) -> ISurfaceApertureCircular:

        s = self.surf.main_row.ApertureData.CurrentTypeSettings

        if self.is_obscuration:
            return s._S_CircularObscuration

        else:
            return s._S_CircularAperture

    @settings.setter
    def settings(self, val: ISurfaceApertureCircular) -> None:
        ZmxAperture.settings.fset(self, val)

    @property
    def min_radius(self) -> u.Quantity:

        return self.settings.MinimumRadius * self.surf.sys.lens_units

    @min_radius.setter
    def min_radius(self, val: u.Quantity):

        s = self.settings
        s.MinimumRadius = val.to(self.surf.sys.lens_units).value
        self.settings = s

    @property
    def max_radius(self) -> u.Quantity:

        return self.settings.MaximumRadius * self.surf.sys.lens_units

    @max_radius.setter
    def max_radius(self, val: u.Quantity):

        s = self.settings
        s.MaximumRadius = val.to(self.surf.sys.lens_units).value
        self.settings = s


class Spider(ZmxAperture, optics.Spider):

    def __init__(self, arm_width: u.Quantity, num_arms: int, surf: 'optics.ZmxSurface'):

        ZmxAperture.__init__(self, surf)

        optics.Spider.__init__(self, arm_width, num_arms)

    @property
    def aperture_type(self) -> SurfaceApertureTypes:

        return SurfaceApertureTypes.Spider

    @property
    def settings(self) -> ISurfaceApertureSpider:

        s = self.surf.main_row.ApertureData.CurrentTypeSettings

        return s._S_Spider

    @settings.setter
    def settings(self, val: ISurfaceApertureSpider) -> None:
        ZmxAperture.settings.fset(self, val)

    @property
    def arm_width(self) -> u.Quantity:

        return self.settings.WidthOfArms * self.surf.sys.lens_units

    @arm_width.setter
    def arm_width(self, val: u.Quantity):

        s = self.settings
        s.WidthOfArms = val.to(self.surf.sys.lens_units).value
        self.settings = s

    @property
    def num_arms(self) -> int:

        return self.settings.NumberOfArms

    @num_arms.setter
    def num_arms(self, val: int):

        s = self.settings
        s.NumberOfArms = val
        self.settings = s
