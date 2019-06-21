
import astropy.units as u

from kgpy import optics
from kgpy.optics.system.configuration.surface import aperture
from .aperture import Aperture
from kgpy.optics.zemax.ZOSAPI.Editors.LDE import SurfaceApertureTypes, ISurfaceApertureRectangular

__all__ = ['Rectangular']


class Rectangular(Aperture, aperture.Rectangular):

    def __init__(self, half_width_x: u.Quantity, half_width_y: u.Quantity, surf: 'optics.ZmxSurface', 
                 attr_str: str):

        Aperture.__init__(self, surf, attr_str)

        aperture.Rectangular.__init__(self, half_width_x, half_width_y)

    @property
    def aperture_type(self) -> SurfaceApertureTypes:

        if self.is_obscuration:
            return SurfaceApertureTypes.RectangularObscuration

        else:
            return SurfaceApertureTypes.RectangularAperture

    @property
    def settings(self) -> ISurfaceApertureRectangular:

        s = Aperture.settings.fget(self)

        if self.is_obscuration:
            return s._S_RectangularObscuration

        else:
            return s._S_RectangularAperture

    @settings.setter
    def settings(self, val: ISurfaceApertureRectangular) -> None:
        Aperture.settings.fset(self, val)

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
