import matplotlib.pyplot as plt
import astropy.time
from kgpy import plot
from . import Slice, fdm


class TestSlice:

    def test_from_time_range(self, capsys):

        with capsys.disabled():

            cds_slice = Slice.from_time_range(
                time_start=astropy.time.Time(fdm.index[0][0]),
                time_end=astropy.time.Time(fdm.index[0][1]),
            )

            plot.CubeSlicer(cds_slice.intensity[:, 0])
