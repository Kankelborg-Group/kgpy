import matplotlib.pyplot as plt
import time
import pytest
import cProfile
import numpy as np
import mayavi.mlab
from . import rebin


def test_rebin():
    x = np.random.rand(2, 5, 7, 3)
    scales = (2, 2, 2, 2)
    new_x = rebin(x, scales)
    assert new_x.shape == tuple(x.shape * np.array(scales))

