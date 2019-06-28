
import typing as tp
import astropy.units as u

from kgpy import optics
from kgpy.optics.zemax import ZOSAPI

from . import Aperture

__all__ = ['Rectangular']


class Rectangular(Aperture, optics.system.configuration.surface.aperture.Rectangular):

    def __init__(self):

        super().__init__()

        self.aperture_type = ZOSAPI.Editors.LDE.SurfaceApertureTypes.RectangularAperture

    @property
    def settings(self) -> tp.Optional[ZOSAPI.Editors.LDE.ISurfaceApertureRectangular]:
        return super().settings

    @settings.setter
    def settings(self, value: ZOSAPI.Editors.LDE.ISurfaceApertureRectangular):
        super().settings = value

    @property
    def half_width_x(self) -> u.Quantity:

        return super().half_width_x

    @half_width_x.setter
    def half_width_x(self, value: u.Quantity):

        super().half_width_x = value

        s = self.settings
        s.XHalfWidth = value.to(self.surf.sys.lens_units).value
        self.settings = s

    @property
    def half_width_y(self) -> u.Quantity:

        return super().half_width_y

    @half_width_y.setter
    def half_width_y(self, value: u.Quantity):

        super().half_width_y = value

        s = self.settings
        s.YHalfWidth = value.to(self.surf.sys.lens_units).value
        self.settings = s

    @property
    def is_obscuration(self) -> bool:
        return super().is_obscuration

    @is_obscuration.setter
    def is_obscuration(self, value: bool):
        super().is_obscuration = value

        try:

            if value:
                self.settings = self.settings._S_RectangularObscuration

            else:
                self.settings = self.settings._S_RectangularAperture

        except AttributeError:
            pass

    def update(self):

        super().update()

        self.half_width_x = self.half_width_x
        self.half_width_y = self.half_width_y
