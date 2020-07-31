import numpy as np
from . import ImageSlicer
import matplotlib.pyplot as plt


class TestImageSlicer:

    def test__init__(self):

        sh = (10, 100)
        y = np.random.random(sh)
        x = np.arange(sh[~0])

        c = ImageSlicer(x[None, ...], y)
