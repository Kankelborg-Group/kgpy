import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

import kgpy.fft.freq
import kgpy.fft.power
from kgpy.fft import random

if __name__ == '__main__':
    angle = 43
    nx = 20
    ny = nx
    nfreq = 4

    img = random.power_law_image(nx, ny, 0, 100, 2)
    k, _, _, _, _ = kgpy.fft.freq.k_arr2d(nx, ny)
    print(k)

    spec_img = kgpy.fft.power.spec2d(img, nfreq, retain_dc=False)

    rot = scipy.ndimage.rotate(img, angle=angle, reshape=False, prefilter=True, order=3)
    rot = scipy.ndimage.rotate(rot, angle=-angle, reshape=False, prefilter=True, order=3)
    spec_rot = kgpy.fft.power.spec2d(rot, nfreq, retain_dc=False)

    # ToDo, convert this to histogram

    plt.imshow(spec_rot)
    plt.show()

