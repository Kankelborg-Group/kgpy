
from kgpy import optics
from kgpy.optics.system.configuration.surface.material import Material as Base

__all__ = ['Material']


class Material(Base):
    
    def __init__(self, name: str, surf: 'optics.ZmxSurface'):
        
        self.surf = surf
        
        Base.__init__(self, name)
        
        surf.main_row.Material = name 
        

        
    @property
    def name(self) -> str:
        
        return self.surf.main_row.Material
    
    @name.setter
    def name(self, val: str):
        
        self.surf.main_row.Material = val
