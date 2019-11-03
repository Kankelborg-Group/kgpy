
import numpy as np
from rebin import rebin as ndarr_rebin
from ndcube import NDCube

def rebin(cube, factors):

    data = cube.data

    data_new = ndarr_rebin(data, factors, func=np.sum)
    if cube.uncertainty:
        uncertainty_new = ndarr_rebin(cube.uncertainty, factors, func=np.linalg.norm)
    else:
        uncertainty_new = None
    if cube.mask:
        mask_new = ndarr_rebin(cube.mask, factors, func=np.max)
    else:
        mask_new = None



    new_wcs = cube.wcs.deepcopy()
    for dim, sz in enumerate(factors):
        correction = 1 / (factors[dim] * data_new.shape[dim] / data.shape[dim])
        crpix = new_wcs.wcs.crpix
        cdelt = new_wcs.wcs.cdelt
        cdelt[-dim-1] *= factors[dim] * correction
        crpix[-dim - 1] /= factors[dim] * correction
        new_wcs.wcs.cdelt = cdelt
        new_wcs.wcs.crpix = crpix
        new_wcs._naxis[-dim-1] = data_new.shape[dim]


    new_cube = NDCube(data_new, new_wcs, uncertainty=uncertainty_new, mask=mask_new, meta=cube.meta, unit=cube.unit,
                      missing_axis=cube.missing_axis)

    return new_cube
