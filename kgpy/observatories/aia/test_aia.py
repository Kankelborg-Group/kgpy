import pathlib
import datetime
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.time
import kgpy.plot
from . import AIA


class TestAIA:

    def test_from_time_range(self, capsys):
        with capsys.disabled():
            time_start = astropy.time.Time('2019-09-30T00:00:00')
            time_end = time_start + 1000 * u.s
            download_path = path = pathlib.Path(__file__).parent / 'test_jsoc'
            channels = [304] * u.AA
            aia = AIA.from_time_range(time_start, time_end, download_path, channels=channels)

            assert aia.intensity.sum() > 0

            sh = aia.intensity.shape[:2]
            assert aia.time.shape == sh
            assert aia.exposure_length.shape == sh
