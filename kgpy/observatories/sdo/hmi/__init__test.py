import pytest
import pathlib
import numpy as np
import astropy.units as u
import astropy.time
from kgpy.obs import __init__test
from . import HMI


@pytest.fixture(scope='module')
def obs_test() -> HMI:
    time_start = astropy.time.Time('2019-09-30T00:00:00')
    time_end = time_start + 60 * u.s
    download_path = pathlib.Path(__file__).parent / 'test_jsoc'

    return HMI.from_time_range(time_start, time_end, download_path)


@pytest.mark.skip
class TestHMI(__init__test.TestImage):

    def test_from_time_range(self,  obs_test: HMI):

        assert np.nansum(obs_test.intensity) != 0

        sh = obs_test.intensity.shape[:2]
        assert obs_test.time.shape == sh
        assert obs_test.exposure_length.shape == sh
