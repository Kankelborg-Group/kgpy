
import astropy.units as u

__all__ = ['Item']


class Item:
    
    def __init__(self, wavelength: u.Quantity):
        
        self.wavelength = wavelength
        self.weight = 1.0
