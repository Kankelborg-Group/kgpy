import pathlib
import numpy as np
import astropy.io.fits
import matplotlib.pyplot as plt

import kgpy
from kgpy.plot.slice.image_stepper import CubeSlicer
from . import despike

testfile = pathlib.Path(kgpy.__file__).parent.parent / 'data/iris_l2_20150615_072426_3610091469_raster_t000_r00000.fits'


def test_percentile_filter():
    
    data = astropy.io.fits.open(testfile)[4].data
    # data = np.empty((1024, 1024, 1024), dtype=np.float32)

    # print(data.shape)

    ksz = (1, 25, 1)

    fdata = despike.percentile_filter(data, 50, ksz)
    
    c = CubeSlicer(fdata)
    plt.show()


def test_identify():

    x = astropy.io.fits.open(testfile)[4].data
    despike.identify(x)

    plt.show()