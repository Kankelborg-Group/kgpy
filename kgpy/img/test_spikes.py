import pathlib
import astropy.io.fits
import matplotlib.pyplot as plt

import kgpy
from kgpy.plot.slice.image_stepper import CubeSlicer
from kgpy.img import spikes

testfile = pathlib.Path(kgpy.__file__).parent.parent / 'data/iris_l2_20150615_072426_3610091469_raster_t000_r00000.fits'


def test_identify_and_fix():

    data = astropy.io.fits.open(testfile)[4].data
    data[data == -200] = 0

    num_frames = None
    data = data[:num_frames]

    fixed_data, mask = spikes.identify_and_fix(data, kernel_size=(11, 11, 21), percentile_threshold=99,
                                               plot_histograms=True)

    c1 = CubeSlicer(data)
    c2 = CubeSlicer(fixed_data)
    c3 = CubeSlicer(mask)

    plt.show()


def test_identify():

    data = astropy.io.fits.open(testfile)[4].data
    data[data == -200] = 0

    num_frames = None
    data = data[:num_frames]

    mask = spikes.identify(data, kernel_size=(11, 11, 21), percentile_threshold=99, plot_histograms=True)

    data[mask == 3] = 0

    c1 = CubeSlicer(data)
    c2 = CubeSlicer(mask)

    plt.show()
