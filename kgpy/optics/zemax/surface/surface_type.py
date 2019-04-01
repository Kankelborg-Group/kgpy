
from abc import ABC, abstractmethod
from win32com.client import CastTo
import astropy.units as u

from kgpy import optics
from kgpy.optics.surface import surface_type
from kgpy.optics.zemax.ZOSAPI.Editors.LDE import SurfaceType, ISurfaceTypeSettings, ISurfaceToroidal, \
    ISurfaceDiffractionGrating

__all__ = []


class ZmxSurfaceType(ABC):

    def __init__(self, surf: 'optics.ZmxSurface'):

        self.surf = surf

        self.settings = self.surf.main_row.GetSurfaceTypeSettings(self.surface_type)

    @property
    @abstractmethod
    def surface_type(self) -> SurfaceType:
        pass

    @property
    def settings(self) -> ISurfaceTypeSettings:

        return self.surf.main_row.CurrentTypeSettings

    @settings.setter
    def settings(self, val: ISurfaceTypeSettings) -> None:

        self.surf.main_row.ChangeType(val)

    @property
    @abstractmethod
    def surface_data(self):

        return self.surf.main_row.SurfaceData


class Toroidal(ZmxSurfaceType, surface_type.Toroidal):

    def __init__(self, radius_of_rotation: u.Quantity, surf: 'optics.ZmxSurface'):

        ZmxSurfaceType.__init__(self, surf)

        surface_type.Toroidal.__init__(self, radius_of_rotation)

    @property
    def surface_type(self):
        return SurfaceType.Toroidal

    @property
    def surface_data(self) -> ISurfaceToroidal:

        return CastTo(self.surf.main_row.SurfaceData, 'ISurfaceToroidal')

    @property
    def radius_of_rotation(self) -> u.Quantity:
        return self.surface_data.RadiusOfRotation * self.surf.sys.lens_units

    @radius_of_rotation.setter
    def radius_of_rotation(self, val: u.Quantity) -> None:

        self.surface_data.RadiusOfRotation = val.to(self.surf.sys.lens_units).value


class DiffractionGrating(ZmxSurfaceType, surface_type.DiffractionGrating):

    def __init__(self, diffraction_order: int, groove_frequency: u.Quantity, surf: 'optics.ZmxSurface'):
        ZmxSurfaceType.__init__(self, surf)

        surface_type.DiffractionGrating.__init__(self, diffraction_order, groove_frequency)

    @property
    def surface_type(self):
        return SurfaceType.DiffractionGrating

    @property
    def surface_data(self) -> ISurfaceDiffractionGrating:
        return CastTo(self.surf.main_row.SurfaceData, 'ISurfaceDiffractionGrating')

    @property
    def groove_frequency(self) -> u.Quantity:

        return self.surface_data.LinesPerMicrometer * (1 / u.um)

    @groove_frequency.setter
    def groove_frequency(self, val: u.Quantity) -> None:

        self.surface_data.LinesPerMicrometer = val.to(1 / u.um).value

    @property
    def diffraction_order(self) -> int:

        return self.surface_data.DiffractionOrder

    @diffraction_order.setter
    def diffraction_order(self, val: int) -> None:

        self.surface_data.DiffractionOrder = val
