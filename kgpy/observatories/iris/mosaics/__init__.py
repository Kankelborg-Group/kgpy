import typing as typ
import numpy as np
import astropy.units as u
import astropy.time
import astropy.wcs
import astropy.io.fits
from kgpy import obs

__all__ = ['Cube', 'saa', 'url']


class Cube(obs.spectral.Cube):

    @classmethod
    def from_path_array(
            cls,
            path_array: np.ndarray,
    ) -> 'Cube':

        hdu_sample = astropy.io.fits.open(str(path_array[0, 0]))[0]

        data_shape = hdu_sample.data.shape[0], 1000, 1000
        rebin_factor = np.array(data_shape) // [data_shape[0], 1000, 1000]
        base_shape = tuple(np.array(data_shape) // rebin_factor)

        self = cls.zeros(path_array.shape + base_shape)
        self.channel = self.channel.value << u.AA

        rw, ry, rx = rebin_factor
        bw, by, bx = base_shape
        qw, qy, qx = base_shape * rebin_factor

        for i in range(path_array.shape[0]):
            for c in range(path_array.shape[1]):
                hdu = astropy.io.fits.open(str(path_array[i, c]))[0]

                d = hdu.data * u.adu
                print(d.shape)
                d = d[:qw, :qy, :qx]
                print(d.shape)
                sw = (bw, rw)
                sx = (bx, rx)
                sy = (by, ry)
                d = d.reshape(sw + sy + sx)
                d = d.sum((~4, ~2, ~0))
                self.intensity[i, c] = d
                self.time[i, c] = astropy.time.Time(hdu.header['DATE_OBS'])
                self.exposure_length[i, c] = float(hdu.header['EXPTIME']) * u.s
                if i == 0:
                    self.channel[c] = float(hdu.header['LAMREF']) * u.AA

        return self


def load_index():

    path_array = np.array([[url.base / path] for path in url.path_list])
    return Cube.from_path_array(path_array)


from . import saa
from . import url
from .mosaics import *
from . import line_profile_moments
