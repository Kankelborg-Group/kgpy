import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

from kgpy.fft import powerspec

if __name__ == '__main__':
    angle = 43
    nx = 2048
    ny = nx
    nfreq = 64

    img = powerspec.random_image(nx, ny, 0, 100, 1)
    k, _, _, _, _ = powerspec.k_arr2d(nx, ny)

    spec_img = powerspec.powerspec2d(img, nfreq, retain_dc=False)

    rot = scipy.ndimage.rotate(img, angle=angle, reshape=False, prefilter=True, order=3)
    rot = scipy.ndimage.rotate(rot, angle=-angle, reshape=False, prefilter=True, order=3)
    spec_rot = powerspec.powerspec2d(rot, nfreq, retain_dc=False)

    # ToDo, convert this to histogram

    plt.imshow(spec_rot - spec_img)
    plt.show()

