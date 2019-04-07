
from kgpy import optics
from kgpy.optics.surface.material import Material as Base

__all__ = ['Material']


class Material(Base):
    
    def __init__(self, name: str, surf: 'optics.ZmxSurface'):
        
        Base.__init__(self, name)
        
        surf.main_row.Material = name 
        
        self.surf = surf
        
    @property
    def name(self) -> str:
        
        return self.surf.main_row.Material
    
    @name.setter
    def name(self, val: str):
        
        self.surf.main_row.Material = val
