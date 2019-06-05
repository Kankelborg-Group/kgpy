
import typing as t
import astropy.units as u

__all__ = ['Spectrograph']


class Spectrograph:

    def __init__(self, wavl_min: u.Quantity, wavl_max: u.Quantity):

        self.wavl_min = wavl_min
        self.wavl_max = wavl_max
