
import astropy.units as u

__all__ = ['Wavelength']


class Wavelength:
    
    def __init__(self, wavelength: u.Quantity, weight: float = 1.0):

        self._wavelength = wavelength
        self._weight = weight

    @property
    def wavelength(self) -> u.Quantity:
        return self._wavelength

    @property
    def weight(self) -> float:
        return self._weight
