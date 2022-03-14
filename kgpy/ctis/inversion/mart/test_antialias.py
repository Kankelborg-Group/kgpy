import numpy as np
import matplotlib.pyplot as plt
from . import antialias


def test_calc_anti_alias_kernel():
    ndims = 4
    x_axis = ~2
    y_axis = ~1
    k = antialias.calc_kernel(ndims, x_axis, y_axis)

    assert k.ndim == ndims
    assert k.sum() == 1.
    assert k.shape[x_axis] == 3
    assert k.shape[y_axis] == 3


def test_apply_anti_aliasing():
    n = 16
    d = np.zeros((1, n, n, 1))
    px = 0, n//2, n//2, 0
    d[px] = 1
    f = antialias.apply(d)
    assert f[px] < 1
    assert f[px] > 0
    assert d.sum() == f.sum()