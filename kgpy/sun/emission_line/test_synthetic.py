import pytest
import matplotlib.pyplot as plt
from kgpy.sun.emission_line.synthetic import synthetic


def test_synthetic():

    A = 1
    x0 = 0
    w = 1
    N = 100

    x, y = synthetic(A, x0, w, N)

    plt.plot(x, y)
    plt.show()

