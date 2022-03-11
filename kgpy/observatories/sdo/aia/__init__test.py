import pytest
import pathlib
import astropy.units as u
import astropy.time
from kgpy.obs import __init__test
from . import AIA


@pytest.fixture(scope='module')
def obs_test() -> AIA:
    time_start = astropy.time.Time('2019-09-30T00:00:00')
    time_end = time_start + 30 * u.s
    download_path = pathlib.Path(__file__).parent / 'test_jsoc'
    channels = [171, 304] * u.AA
    return AIA.from_time_range(time_start, time_end, download_path, channels=channels)


class TestAIA(__init__test.TestImage):

    def test_from_time_range(self,  obs_test: AIA):

        assert obs_test.intensity.sum() > 0

        sh = obs_test.intensity.shape[:2]
        assert obs_test.time.shape == sh
        assert obs_test.exposure_length.shape == sh
