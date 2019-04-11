
import astropy.units as u

from kgpy import optics

__all__ = ['Material']


class Material:
    
    def __init__(self, name: str):
        
        self.name = name

    def promote_to_zmx(self, surf: 'optics.ZmxSurface'):
        
        m = optics.zemax.surface.material.Material(self.name, surf)
        
        return m

class Mirror(Material):
    
    def __init__(self):
        
        Material.__init__(self, 'MIRROR')


class NoMaterial(Material):

    def __init__(self):
        
        Material.__init__(self, '')
