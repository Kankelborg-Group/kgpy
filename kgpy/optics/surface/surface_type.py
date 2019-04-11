
import astropy.units as u

from kgpy import optics

__all__ = ['SurfaceType', 'Standard', 'Paraxial', 'Toroidal', 'DiffractionGrating', 'EllipticalGrating1']


class SurfaceType:

    def promote_to_zmx(self, surf: 'optics.ZmxSurface') -> 'optics.zemax.surface.surface_type.SurfaceType':
        
        raise NotImplementedError()


class Standard(SurfaceType):

    def __init__(self):
        SurfaceType.__init__(self)

    def promote_to_zmx(self, surf: 'optics.ZmxSurface'):

        s = optics.zemax.surface.surface_type.Standard(surf)

        return s


class Paraxial(SurfaceType):

    def __init__(self):
        SurfaceType.__init__(self)


class Toroidal(SurfaceType):

    def __init__(self, radius_of_rotation: u.Quantity = 1 * u.mm):

        SurfaceType.__init__(self)

        self.radius_of_rotation = radius_of_rotation


class DiffractionGrating(SurfaceType):

    def __init__(self, diffraction_order: int = 1, groove_frequency: u.Quantity = 1 / u.mm):

        SurfaceType.__init__(self)

        self.diffraction_order = diffraction_order
        self.groove_frequency = groove_frequency


class EllipticalGrating1(SurfaceType):

    def __init__(self):

        SurfaceType.__init__(self)

        self.groove_frequency = 0 / u.um
        self.diffraction_order = 0.0
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0 * u.mm
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0
        self.delta = 0.0
        self.epsilon = 0.0
        self.max_terms = 0
        self.norm_radius = 100.0

    def promote_to_zmx(self, surf: 'optics.ZmxSurface') -> 'optics.zemax.surface.surface_type.EllipticalGrating1':

        s = optics.zemax.surface.surface_type.EllipticalGrating1(surf)

        s.groove_frequency = self.groove_frequency
        s.diffraction_order = self.diffraction_order
        s.a = self.a
        s.b = self.b
        s.c = self.c
        s.alpha = self.alpha
        s.beta = self.beta
        s.gamma = self.gamma
        s.delta = self.delta
        s.epsilon = self.epsilon
        s.norm_radius = self.norm_radius

        return s
