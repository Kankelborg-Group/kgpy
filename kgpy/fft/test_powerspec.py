import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib

import kgpy.fft.freq
import kgpy.fft.power
from kgpy.fft import random

if __name__ == '__main__':
    angle = 22
    nx = 4096
    ny = nx
    nfreq = 128

    cval=0

    img = powerspec.random_image(nx, ny, 0, 100, 2)

    border = int(np.ceil((1 / 2) * (nx) * (1 - np.sqrt(2) / 2))) + 5
    img[:border, :,] = cval
    img[~border:, :,] = cval
    img[:, :border,] = cval
    img[:, ~border:,] = cval

    k, _, _, _, _ = powerspec.k_arr2d(nx, ny)

    spec_img = powerspec.powerspec2d(img[border:~border, border:~border], nfreq, retain_dc=False)

    rot_kwargs = {
        'reshape': False,
        'prefilter': True,
        'order': 5,
        'mode': 'nearest'
    }

    rot = scipy.ndimage.rotate(img, angle=angle, **rot_kwargs)
    rot = scipy.ndimage.rotate(rot, angle=-angle, **rot_kwargs)
    spec_rot = powerspec.powerspec2d(rot[border:~border, border:~border], nfreq, retain_dc=False)

    # ToDo, convert this to histogram

    coordfig, coordax = plt.subplots(1, 3, constrained_layout=True)
    cim0 = coordax[0].imshow(img)
    coordfig.colorbar(cim0, ax=coordax[0], shrink=0.4)
    cim1 = coordax[1].imshow(rot)
    coordfig.colorbar(cim0, ax=coordax[1], shrink=0.4)
    cim2 = coordax[2].imshow(rot - img)
    coordfig.colorbar(cim2, ax=coordax[2], shrink=0.4)
    for j, title in enumerate(['original', 'rotated', 'residual']):
        coordax[j].set_title(title)

    spec_ratio = spec_rot / spec_img

    kfig, kax = plt.subplots(1, 3, constrained_layout=True)
    kim0 = kax[0].imshow(spec_img)
    kfig.colorbar(kim0, ax=kax[0], shrink=0.4,)
    kim1 = kax[1].imshow(spec_rot)
    kfig.colorbar(kim0, ax=kax[1], shrink=0.4)
    kim2 = kax[2].imshow(spec_ratio)
    kfig.colorbar(kim2, ax=kax[2], shrink=.4)
    for j, title in enumerate(['P(original)', 'P(rotated)', 'P(rot) / P(orig)']):
        kax[j].set_title(title)

    kernl = np.sqrt(np.abs(scipy.fft.fftn(spec_ratio)))
    kernl = np.roll(kernl, (16, 16), axis=(0,1))
    plt.figure()
    plt.imshow(kernl, norm=matplotlib.colors.LogNorm(1))
    plt.show()
