import pathlib
import numpy as np
import astropy.io.fits
import matplotlib.pyplot as plt

import kgpy
from kgpy.plot.slice.image_stepper import CubeSlicer
from . import despike

testfile = pathlib.Path(kgpy.__file__).parent.parent / 'data/iris_l2_20150615_072426_3610091469_raster_t000_r00000.fits'


def test_identify():

    x = astropy.io.fits.open(testfile)[4].data
    despike.identify(x)

    plt.show()