import dataclasses
import astropy.units as u

__all__ = ['Wavelengths']


@dataclasses.dataclass
class Wavelengths:
    
    wavelengths: u.Quantity
    weights: u.Quantity = 1.0 * u.dimensionless_unscaled
