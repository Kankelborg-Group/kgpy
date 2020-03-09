import pytest
import numpy as np

from .slab_array import SlabArray


class TestSlabArray:

    @pytest.fixture
    def slab_array(self) -> SlabArray:
        n_slabs = 3
        data = []
        for i in range(n_slabs):
            data.append(np.ones((10, 10)))
        return SlabArray(data)

    def test__neg___(self, slab_array):
        a = -slab_array
        assert a.data[0].sum() < 0


    def test__add__(self, slab_array):
        a = -slab_array
        b = slab_array.copy()
        c = a + b
        assert c.data[0].sum() == 0

    def test__sub__(self, slab_array):
        a = slab_array.copy()
        b = slab_array.copy()
        c = a - b
        assert c.data[0].sum() == 0

    def test__mul__(self, slab_array):
        a = -slab_array
        b = slab_array.copy()
        c = a * b
        assert c.data[0].sum() < -0

    def test__truediv__(self, slab_array):
        a = -slab_array
        b = slab_array.copy()
        c = a / b
        assert c.data[0][0, 0] == -1
