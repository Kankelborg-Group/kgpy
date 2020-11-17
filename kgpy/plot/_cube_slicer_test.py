import numpy as np
from . import CubeSlicer


class TestCubeSlicer:

    def test__init__(self):
        cube = np.random.rand(20, 20, 40)
        cs = CubeSlicer(cube)
        assert isinstance(cs, CubeSlicer)
