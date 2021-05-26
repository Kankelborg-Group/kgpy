"""
N-dimensional image processing package
"""
__all__ = ['spikes']

from . import spikes


import numpy as np
import scipy.signal

def event_selector(img, pix, threshold_int):
    mask = np.zeros_like(img, dtype=bool)
    mask[pix] = True
    not_converged = True
    kernel = np.ones((3, 3))
    final_masked_im = np.copy(img)
    while not_converged:
        new_mask = scipy.signal.convolve2d(mask, kernel, mode='same')

        mask[new_mask > 0] = True

        masked_im = np.copy(img)
        masked_im[~mask] = 0
        mask[masked_im < threshold_int] = False
        if np.sum(final_masked_im - masked_im) == 0:
            not_converged = False
        else:
            final_masked_im = np.copy(masked_im)

    return mask