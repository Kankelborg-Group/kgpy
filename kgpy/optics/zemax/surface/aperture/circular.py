
import astropy.units as u

from kgpy import optics
from kgpy.optics.surface import aperture
from .aperture import Aperture
from kgpy.optics.zemax.ZOSAPI.Editors.LDE import SurfaceApertureTypes, ISurfaceApertureCircular

__all__ = ['Circular']


class Circular(Aperture, aperture.Circular):

    def __init__(self, min_radius: u.Quantity, max_radius: u.Quantity, surf: 'optics.ZmxSurface'):

        Aperture.__init__(self, surf)

        aperture.Circular.__init__(self, min_radius, max_radius)

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
        Aperture.settings.fset(self, val)

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
