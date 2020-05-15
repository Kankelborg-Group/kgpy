import numpy as np
import astropy.units as u
from . import Aperture

__all__ = ['NoAperture']


class NoAperture(Aperture):

    def to_zemax(self) -> 'NoAperture':
        raise NotImplementedError

    def is_unvignetted(self, points: u.Quantity) -> np.ndarray:
        return np.array(True)

