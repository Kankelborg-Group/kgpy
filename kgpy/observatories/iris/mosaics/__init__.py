import typing as typ
import pathlib
import numpy as np
import astropy.units as u
import astropy.time
import astropy.wcs
import astropy.io.fits
from kgpy import obs, img
from . import saa
from . import url

__all__ = ['Cube', 'saa', 'url']


class Cube(obs.spectral.Cube):

    @classmethod
    def from_path_array(
            cls,
            path_array: np.ndarray,
    ) -> 'Cube':

        hdu_sample = astropy.io.fits.open(str(path_array[0, 0]))[0]

        spatial_shape = 1000, 1000

        base_shape = spatial_shape + hdu_sample.data.shape[:1]
        self = cls.zeros(path_array.shape + base_shape)
        self.channel = self.channel.value << u.AA

        for i in range(path_array.shape[0]):
            for c in range(path_array.shape[1]):
                print(path_array[i, c])

                hdu = astropy.io.fits.open(str(path_array[i, c]))[0]

                d = hdu.data * u.adu

                rebin_factor = np.array(d.shape[~1:]) // np.array(spatial_shape)

                ry, rx = rebin_factor
                by, bx = spatial_shape
                qy, qx = spatial_shape * rebin_factor

                d = d[..., :qy, :qx]
                sx = (bx, rx)
                sy = (by, ry)
                d = d.reshape((-1, ) + sy + sx)
                d = d.sum((~2, ~0))

                for region_x, region_y in zip(saa.regions_x[i], saa.regions_y[i]):
                    d[..., slice(*region_y), slice(*region_x)] = 0

                d = img.spikes.identify_and_fix(d.value, percentile_threshold=99.9, kernel_size=5)[0] << d.unit

                self.intensity[i, c] = np.moveaxis(d, 0, ~0)
                self.time[i, c] = astropy.time.Time(hdu.header['DATE_OBS'])
                self.exposure_length[i, c] = float(hdu.header['EXPTIME']) * u.s
                if i == 0:
                    self.channel[c] = float(hdu.header['LAMREF']) * u.AA

                wcs = astropy.wcs.WCS(hdu.header)
                wcs.wcs.cdelt[~1:] *= rebin_factor
                wcs.wcs.crpix[~1:] /= rebin_factor
                wcs = wcs.swapaxes(~1, ~0)
                wcs = wcs.swapaxes(~2, ~1)
                wcs.array_shape = base_shape
                self.wcs[i, c] = wcs

        return self


index_cache = pathlib.Path(__file__).parent / 'index_cache.pickle'


def load_index(disk_cache: pathlib.Path = index_cache):

    if disk_cache.exists():
        cube = Cube.from_pickle(disk_cache)

    else:

        path_array = np.array([[url.base / path] for path in url.path_list])

        # path_array = path_array[:4]

        cube = Cube.from_path_array(path_array)

        # for i in range(path_array.shape[0]):
        #     for region_x, region_y in zip(saa.regions_x[i], saa.regions_y[i]):
        #         cube.intensity[i, ..., slice(*region_y), slice(*region_x), :] = 0

        cube.to_pickle(disk_cache)

    return cube


from .mosaics import *
from . import line_profile_moments
