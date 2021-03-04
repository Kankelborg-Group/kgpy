import pytest
import pathlib
import datetime
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.time
import kgpy.plot
# from .. import __init__test
from . import HMI


# @pytest.fixture(scope='module')
# def obs_test() -> HMI:
#     time_start = astropy.time.Time('2019-09-30T00:00:00')
#     time_end = time_start + 60 * u.s
#     download_path = pathlib.Path(__file__).parent / 'test_jsoc'
#
#     return HMI.from_time_range(time_start, time_end, download_path)


# class TestHMI(__init__test.TestObs):
#
#     def test_from_time_range(self,  obs_test: HMI):
#         with capsys.disabled():
#             time_start = astropy.time.Time('2019-09-30T18:08:00')
#             time_end = time_start + 100 * u.s
#             download_path = path = pathlib.Path(__file__).parent / 'test_jsoc'
#
#             hmi = HMI.from_time_range(time_start, time_end, download_path)
#
#         c = kgpy.plot.CubeSlicer(obs_test.intensity[:, 0].value)
#         plt.show()
#
#         assert obs_test.intensity.sum() > 0
#
#         sh = obs_test.intensity.shape[:2]
#         assert obs_test.time.shape == sh
#         assert obs_test.exposure_length.shape == sh

def test_hmi():
    time_start = astropy.time.Time('2019-09-30T18:00:00')
    time_end = time_start + 20 * u.min
    download_path = path = pathlib.Path(__file__).parent / 'test_jsoc'

    hmi = HMI.from_time_range(time_start, time_end, download_path)
    rails = 100
    print(hmi.intensity.shape)
    c = kgpy.plot.CubeSlicer(hmi.intensity.value,vmin=-rails,vmax=rails)
    plt.show()