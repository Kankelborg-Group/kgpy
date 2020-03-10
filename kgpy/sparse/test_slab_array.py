import pytest
import numpy as np

from .slab_array import SlabArray


class TestSlabArray:
    n_slabs = 3
    nx = 10
    ny = 10
    nz = 5

    @pytest.fixture
    def slab_array(self) -> SlabArray:
        data = []
        for i in range(self.n_slabs):
            data.append(np.ones((self.nx, self.ny, self.nz)))
        return SlabArray(data, 7 * np.arange(self.n_slabs))

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

    def test_sum(self, slab_array):
        c = slab_array.copy()
        assert c.sum() == self.n_slabs * self.nx * self.ny * self.nz