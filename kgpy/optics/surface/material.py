
import astropy.units as u

__all__ = ['Material']


class Material:
    
    def __init__(self, name: str):
        
        self.name = name


class Mirror(Material):
    
    def __init__(self):
        
        Material.__init__(self, 'MIRROR')


class NoMaterial(Material):

    def __init__(self):
        
        Material.__init__(self, '')
