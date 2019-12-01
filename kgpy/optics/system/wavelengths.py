import dataclasses
import numpy as np
import astropy.units as u

__all__ = ['Wavelengths']


@dataclasses.dataclass
class Wavelengths:
    
    values: u.Quantity = dataclasses.field(default_factory=lambda: [533] * u.nm)
    weights: u.Quantity = dataclasses.field(default_factory=lambda: [1.0] * u.dimensionless_unscaled)

    @property
    def config_broadcast(self):
        return np.broadcast(
            self.values[..., 0],
            self.weights[..., 0],
        )
    
    @property
    def num_per_config(self):
        return np.broadcast(self.values, self.weights).shape[~0]
