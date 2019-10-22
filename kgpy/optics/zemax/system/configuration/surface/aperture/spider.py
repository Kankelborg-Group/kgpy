
import astropy.units as u

from kgpy import optics
from kgpy.optics.system.surface_array import aperture
from .aperture import Aperture
from kgpy.optics.zemax.ZOSAPI.Editors.LDE import SurfaceApertureTypes, ISurfaceApertureSpider

__all__ = ['Spider']


class Spider(Aperture, aperture.Spider):

    def __init__(self, arm_width: u.Quantity, num_arms: int, surf: 'optics.ZmxSurface',
                 attr_str: str):

        Aperture.__init__(self, surf, attr_str)

        aperture.Spider.__init__(self, arm_width, num_arms)

    @property
    def aperture_type(self) -> SurfaceApertureTypes:

        return SurfaceApertureTypes.Spider

    @property
    def settings(self) -> ISurfaceApertureSpider:

        s = Aperture.settings.fget(self)

        return s._S_Spider

    @settings.setter
    def settings(self, value: ISurfaceApertureSpider) -> None:
        Aperture.settings.fset(self, value)

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
