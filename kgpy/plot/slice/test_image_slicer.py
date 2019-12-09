import numpy as np
from . import ImageSlicer


class TestImageSlicer:

    def test__init__(self):

        d = np.random.random((10, 100))

        c = ImageSlicer()