import pathlib
import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
import astropy.wcs
import astropy.io.fits
import kgpy.obs

__all__ = ['Image']


@dataclasses.dataclass
class Image(kgpy.obs.Image):

    @classmethod
    def from_path_array(
            cls,
            path_array: typ.Sequence[pathlib.Path],
    ) -> 'Image':

        for c, path in enumerate(path_array):
            # print(path)

            hdu_list = astropy.io.fits.open(str(path))
            hdu = hdu_list[0]

            if c == 0:
                base_shape = hdu.data.shape
                self = cls.zeros(base_shape[:1] + (len(path_array),) + base_shape[1:])
                self.channel = self.channel.value << u.AA

            d = hdu.data * u.adu

            # print(repr(hdu.header))
            # print(repr(hdu_list[~1].header))

            self.intensity[:, c] = d

            self.time[:, c] = astropy.time.Time(hdu.header['DATE_OBS']) + hdu_list[1].data[..., 0] * u.s
            self.exposure_length[:, c] = float(hdu.header['EXPTIME']) * u.s
            self.channel[c] = float(hdu.header['TWAVE1']) * u.AA

            xcen = (hdu_list[~1].data[..., hdu_list[~1].header['XCENIX']] * u.arcsec).to(u.deg)
            ycen = (hdu_list[~1].data[..., hdu_list[~1].header['YCENIX']] * u.arcsec).to(u.deg)
            for i in range(self.shape[self.axis.time]):
                wcs = astropy.wcs.WCS(hdu.header)
                if xcen[i].value != 0:
                    wcs.wcs.crval[0] = xcen[i].value
                if ycen[i].value != 0:
                    wcs.wcs.crval[1] = ycen[i].value
                wcs.wcs.set()
                self.wcs[i, c] = wcs

        self.intensity[self.intensity == -200 * u.adu] = np.nan

        return self
