import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import astropy.units as u
import astropy.time
import astropy.wcs
from . import Obs

__all__ = ['TestObs']


@pytest.fixture
def obs_test() -> Obs:
    shape = (10, 3, 128, 256)
    sh = shape[:2]

    wcs = astropy.wcs.WCS(naxis=2)
    wcs.array_shape = shape[2:]
    wcs.wcs.ctype = ['X', 'Y']
    wcs_arr = np.empty(sh, dtype=astropy.wcs.WCS)
    wcs_arr[:] = wcs

    return Obs(
        intensity=100 * np.random.random(shape) * u.adu,
        intensity_uncertainty=10*u.adu,
        wcs=wcs_arr,
        time=astropy.time.Time.now() + np.broadcast_to(np.arange(shape[0])[..., None] * u.s, sh),
        time_index=np.arange(shape[0]),
        channel=(np.arange(shape[1]) + 1) * u.chan,
        exposure_length=np.ones(sh) * u.s
    )


class TestObs:

    def test_zeros(self):
        shape = (5, 7, 11, 13)
        sh = shape[:2]
        obs = Obs.zeros(shape)
        assert obs.intensity.shape == shape
        assert obs.intensity_uncertainty.shape == shape
        assert obs.wcs.shape == sh
        assert obs.time.shape == sh
        assert obs.time_index.shape == shape[0:1]
        assert obs.channel.shape == shape[1:2]
        assert obs.exposure_length.shape == sh

        assert obs.intensity.sum() == 0
        assert obs.intensity_uncertainty.sum() == 0

    def test_shape(self, obs_test):
        assert len(obs_test.shape) == 4

    def test_num_times(self, obs_test):
        assert isinstance(obs_test.num_times, int)

    def test_num_channels(self, obs_test):
        assert isinstance(obs_test.num_channels, int)

    def test_channel_labels(self, obs_test):
        assert len(obs_test.channel_labels) == obs_test.num_channels

    def test_plot_intensity_total_vs_time(self, obs_test):
        ax = obs_test.plot_intensity_total_vs_time()
        assert ax.lines
        plt.close(ax.figure)

    def test_plot_exposure_length(self, obs_test):
        ax = obs_test.plot_exposure_length()
        assert ax.lines
        plt.close(ax.figure)

    def test_plot_intensity_channel(self, obs_test):
        ax = obs_test.plot_intensity_channel()
        assert ax.images
        plt.close(ax.figure)

    def test_plot_intensity_time(self, obs_test):
        axs = obs_test.plot_intensity_time()
        for ax in axs.flatten():
            assert ax.images
        plt.close(axs[0, 0].figure)

    def test_animate_intensity_channel(self, obs_test):
        anim = obs_test.animate_intensity_channel()
        assert isinstance(anim, matplotlib.animation.FuncAnimation)
        plt.close(anim._fig)

    def test_animate_intensity(self, obs_test):
        anim = obs_test.animate_intensity()
        assert isinstance(anim, matplotlib.animation.FuncAnimation)
        plt.close(anim._fig)
