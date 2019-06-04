
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

    def calc_linear_channels(self) -> t.List[Spectrograph]:

        wavl_range = self.wavl_max - self.wavl_min

        range_per_chan = wavl_range / self.num_channels

        wavl_overlap = range_per_chan * self.overlap_factor

        equiv_range = wavl_range - (self.num_channels - 1) * wavl_overlap

        equiv_range_per_chan = equiv_range / self.num_channels

        channels = []

        # w_min = self.wavl_min
        # w_max = w_min + range_per_chan

        for c in range(self.num_channels):

            w_min = self.wavl_min + c * equiv_range_per_chan
            w_max = w_min + range_per_chan

            s = Spectrograph(w_min, w_max)

            channels.append(s)

        return channels
