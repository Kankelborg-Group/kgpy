
from abc import ABC, abstractmethod
from win32com.client import CastTo
import astropy.units as u

from kgpy import optics
from kgpy.optics.system.surface import surface_type
from kgpy.optics.zemax.ZOSAPI.Editors.LDE import SurfaceType, ISurfaceTypeSettings, ISurfaceToroidal, \
    ISurfaceDiffractionGrating, ISurfaceEllipticalGrating1

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


class Standard(ZmxSurfaceType, surface_type.Standard):
    
    def __init__(self, surf: 'optics.ZmxSurface'):
        
        ZmxSurfaceType.__init__(self, surf)
        
        surface_type.Standard.__init__(self)
        
    @property
    def surface_type(self):
        return SurfaceType.Standard

    @property
    def surface_data(self) -> ISurfaceToroidal:

        return CastTo(self.surf.main_row.SurfaceData, 'ISurfaceStandard')


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

        return self.surface_data.LinesPerMicroMeter * (1 / u.um)

    @groove_frequency.setter
    def groove_frequency(self, val: u.Quantity) -> None:

        self.surface_data.LinesPerMicroMeter = val.to(1 / u.um).value

    @property
    def diffraction_order(self) -> int:

        return self.surface_data.DiffractionOrder

    @diffraction_order.setter
    def diffraction_order(self, val: int) -> None:

        self.surface_data.DiffractionOrder = val


class EllipticalGrating1(ZmxSurfaceType, surface_type.EllipticalGrating1):

    def __init__(self, surf: 'optics.ZmxSurface'):

        ZmxSurfaceType.__init__(self, surf)

        surface_type.EllipticalGrating1.__init__(self)

    @property
    def surface_type(self):
        return SurfaceType.EllipticalGrating1

    @property
    def surface_data(self) -> ISurfaceEllipticalGrating1:
        return CastTo(self.surf.main_row.SurfaceData, 'ISurfaceEllipticalGrating1')

    @property
    def groove_frequency(self) -> u.Quantity:

        return self.surface_data.LinesPerMicroMeter * (1 / u.um)

    @groove_frequency.setter
    def groove_frequency(self, val: u.Quantity) -> None:

        self.surface_data.LinesPerMicroMeter = val.to(1 / u.um).value

    @property
    def diffraction_order(self) -> float:

        return self.surface_data.DiffractionOrder

    @diffraction_order.setter
    def diffraction_order(self, val: float) -> None:

        self.surface_data.DiffractionOrder = val

    @property
    def a(self) -> float:
        return self.surface_data.A

    @a.setter
    def a(self, value: float):
        self.surface_data.A = value

    @property
    def b(self) -> float:
        return self.surface_data.B

    @b.setter
    def b(self, value: float):
        self.surface_data.B = value

    @property
    def c(self) -> u.Quantity:
        return self.surface_data.c * self.surf.sys.lens_units

    @c.setter
    def c(self, value: u.Quantity):
        
        self.surface_data.c = value.to(self.surf.sys.lens_units).value

    @property
    def alpha(self) -> float:
        return self.surface_data.Alpha

    @alpha.setter
    def alpha(self, value: float):
        self.surface_data.Alpha = value
        
    @property
    def beta(self) -> float:
        return self.surface_data.Beta

    @beta.setter
    def beta(self, value: float):
        self.surface_data.Beta = value
        
    @property
    def gamma(self) -> float:
        return self.surface_data.Gamma

    @gamma.setter
    def gamma(self, value: float):
        self.surface_data.Gamma = value
        
    @property
    def delta(self) -> float:
        return self.surface_data.Delta

    @delta.setter
    def delta(self, value: float):
        self.surface_data.Delta = value
        
    @property
    def epsilon(self) -> float:
        return self.surface_data.Epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        self.surface_data.Epsilon = value

    @property
    def norm_radius(self) -> float:
        return self.surface_data.NormRadius

    @norm_radius.setter
    def norm_radius(self, value: float):
        self.surface_data.NormRadius = value
        
    @property
    def num_terms(self) -> int:
        return self.surface_data.NumberOfTerms
    
    @num_terms.setter
    def num_terms(self, value: int):
        self.surface_data.NumberOfTerms = value
