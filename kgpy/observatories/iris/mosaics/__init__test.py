import numpy as np
import matplotlib.pyplot as plt
import kgpy.plot
from . import load_index


def test_load_index(capsys):
    with capsys.disabled():
        c = load_index()

        print(c.intensity.shape)
        hs = kgpy.plot.HypercubeSlicer(
            c.intensity[:, 0].value, wcs_list=c.wcs[:, 0], height_ratios=(5, 1), width_ratios=(5, 1))
        plt.show()