
import typing as tp
import astropy.units as u

from kgpy import optics
from kgpy.optics.zemax import ZOSAPI

from . import Aperture

__all__ = ['Rectangular']


class Rectangular(optics.system.configuration.surface.aperture.Rectangular, Aperture):

    def __init__(self, zemax_surface, *args, **kwargs):

        super().__init__(*args, **kwargs)

        Aperture.__init__(self)

        if self.is_obscuration:
            aperture_type = ZOSAPI.Editors.LDE.SurfaceApertureTypes.RectangularObscuration
        else:
            aperture_type = ZOSAPI.Editors.LDE.SurfaceApertureTypes.RectangularAperture

        settings = zemax_surface.ApertureData.CreateApertureTypeSettings(
            aperture_type)     # type: ZOSAPI.Editors.LDE.ISurfaceApertureRectangular

        settings.ApertureXDecenter = self.decenter_x
        settings.ApertureYDecenter = self.decenter_y
        settings.XHalfWidth = self.half_width_x
        settings.YHalfWidth = self.half_width_y

        zemax_surface.ApertureData.ChangeApertureTypeSettings(settings)
