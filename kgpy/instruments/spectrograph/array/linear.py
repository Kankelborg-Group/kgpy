
import typing as t
import astropy.units as u

from ..spectrograph import Spectrograph
from .array import Array


class Linear(Array):

    def __init__(self, num_channels: int, wavl_min: u.Quantity, wavl_max: u.Quantity, overlap_factor: float):

        self.num_channels = num_channels

        self.wavl_min = wavl_min
        self.wavl_max = wavl_max
        self.overlap_factor = overlap_factor

        channels = self.calc_linear_channels()

        super().__init__(channels)

    def __iter__(self) -> t.Iterator[Spectrograph]:
        return super().__iter__()

    @property
    def data(self) -> t.List[Spectrograph]:
        return self.data

    def calc_linear_channels(self) -> t.List[Spectrograph]:

        pass
