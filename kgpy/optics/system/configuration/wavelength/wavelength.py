
import astropy.units as u

__all__ = ['Wavelength']


class Wavelength:
    
    def __init__(self, wavelength: u.Quantity):
        
        self.wavelength = wavelength
        self.weight = 1.0
