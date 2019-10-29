import dataclasses
import numpy as np
import astropy.units as u

__all__ = ['Wavelengths']


@dataclasses.dataclass
class Wavelengths:
    
    wavelengths: u.Quantity = [533] * u.nm
    weights: u.Quantity = [1.0] * u.dimensionless_unscaled

    @property
    def broadcasted_attrs(self):
        return np.broadcast(
            self.wavelengths[..., 0],
            self.weights[..., 0],
        )