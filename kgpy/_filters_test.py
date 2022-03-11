import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import kgpy.filters


def test_gaussian_trimmed():

    shape = (100, 100)
    a = np.random.normal(loc=100, scale=25, size=shape)

    b = kgpy.filters.gaussian_trimmed(
        array=a,
        kernel_size=41,
        kernel_width=3,
        proportion=35 * u.percent
    )

    assert b.shape == shape
    assert b.sum() != 0

