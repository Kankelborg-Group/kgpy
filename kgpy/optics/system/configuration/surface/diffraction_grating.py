
import astropy.units as u

from . import Standard

__all__ = ['DiffractionGrating']


class DiffractionGrating(Standard):

    def __init__(self, *args, diffraction_order: int, groove_frequency: u.Quantity, **kwargs):

        super().__init__(*args, **kwargs)

        self._diffraction_order = diffraction_order
        self._groove_frequency = groove_frequency

    @property
    def diffraction_order(self) -> int:
        return self._diffraction_order

    @property
    def groove_frequency(self) -> u.Quantity:
        return self._groove_frequency
