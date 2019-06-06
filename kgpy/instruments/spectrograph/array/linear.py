
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

        dl = self.wavl_max - self.wavl_min

        a = dl / (self.num_channels * (1 - self.overlap_factor) - self.overlap_factor)

        b = self.overlap_factor * a

        channels = []

        w_max = self.wavl_min

        for c in range(self.num_channels):

            w_min = w_max - b
            w_max = w_min + a

            channels.append(Spectrograph(w_min, w_max))

        return channels
