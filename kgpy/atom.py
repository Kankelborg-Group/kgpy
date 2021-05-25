import dataclasses
import astropy.units as u

__all__ = ['Transition']


@dataclasses.dataclass
class Transition:
    ion: str
    wavelength: u.Quantity
    intensity: u.Quantity
