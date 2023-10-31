import pytest
import pathlib
import astropy.units as u
import astropy.time
from kgpy.obs import __init__test
from . import RadiantIntensity


@pytest.fixture(scope='module')
def obs_test() -> RadiantIntensity:
    time_start = astropy.time.Time('2019-09-30T00:00:00')
    time_end = time_start + 30 * u.s
    download_path = pathlib.Path(__file__).parent / 'test_jsoc'
    channels = [171, 304] * u.AA
    return RadiantIntensity.from_time_range(time_start, time_end, download_path, channels=channels)


@pytest.mark.skip
class TestRadiantIntensity:

    def test_from_time_range(self,  obs_test: RadiantIntensity):

        assert obs_test.output.sum() > 0
