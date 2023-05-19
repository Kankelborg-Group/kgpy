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


def test_mean_trimmed():

    shape = (100, 100)
    a = np.random.normal(loc=100, scale=25, size=shape)

    b = kgpy.filters.mean_trimmed(
        array=a,
        kernel_size=21,
        proportion=0 * u.percent
    )

    assert b.shape == shape
    assert b.sum() != 0


def test_mean_1d_trimmed():

    shape = (100, 100)
    a = np.random.normal(loc=100, scale=25, size=shape)

    b = kgpy.filters.mean_1d_trimmed(
        array=a,
        kernel_size=21,
        proportion=0.25
    )

    assert b.shape == shape
    assert b.sum() != 0


