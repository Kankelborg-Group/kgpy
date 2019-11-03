
import matplotlib.pyplot as plt

from kgpy.img.dspk import dspk_ndarr
from kgpy.plot.slice.image_stepper import CubeSlicer
from matplotlib import colors

def dspk_3D(data, lower_thresh=0.01, upper_thresh=0.99, kernel_shape=(25,25,25)):

    lmed = dspk_ndarr(data, lower_thresh, upper_thresh, kernel_shape[0], kernel_shape[1], kernel_shape[2],-200)

    return lmed


def plt_dspk(data, results):
    gmap = results[0]
    histogram = results[1]
    t1 = results[2]
    t9 = results[3]
    cnts = results[4]

    dplt = CubeSlicer(data)
    mplt = CubeSlicer(gmap)

    for ind in range(3):
        plt.figure()
        plt.imshow(histogram[ind, :, :], norm=colors.SymLogNorm(1e-4), origin='lower')
        plt.plot(t1[ind, 0, :])
        plt.plot(t9[ind, 0, :])