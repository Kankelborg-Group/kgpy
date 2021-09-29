import pathlib
import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
import astropy.time
import astropy.wcs
import astropy.io.fits
import kgpy.obs
import kgpy.mixin

__all__ = ['Image']


@dataclasses.dataclass
class Image(kgpy.obs.Image):
    time_wcs: typ.Optional[u.Quantity] = None

    @classmethod
    def zeros(cls, shape: typ.Sequence[int]) -> 'Image':
        self = super().zeros(shape=shape)
        self.time_wcs = np.zeros(self.time.shape) * u.s
        return self

    @classmethod
    def from_path_sequence(
            cls,
            path_sequence: typ.Sequence[pathlib.Path],
    ) -> 'Image':

        hdu_list = astropy.io.fits.open(str(path_sequence[0]))
        hdu = hdu_list[0]
        base_shape = hdu.data.shape
        self = cls.zeros(base_shape[:1] + (len(path_sequence),) + base_shape[1:])
        self.channel = self.channel.value << u.AA

        for c, path in enumerate(path_sequence):
            # print(path)

            hdu_list = astropy.io.fits.open(str(path))
            hdu = hdu_list[0]

            d = hdu.data * u.adu

            # print(repr(hdu.header))
            # print(repr(hdu_list[~1].header))

            self.intensity[:, c] = d

            self.time_wcs[:, c] = hdu_list[1].data[..., 0] * u.s
            self.time[:, c] = astropy.time.Time(hdu.header['DATE_OBS']) + self.time_wcs[:, c]
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


@dataclasses.dataclass
class ImageList(kgpy.mixin.DataclassList[Image]):

    def to_image(self) -> Image:
        def concat(param: str):
            return np.concatenate([getattr(img, param) for img in self])
        return Image(
            intensity=concat('intensity'),
            intensity_uncertainty=concat('intensity_uncertainty'),
            wcs=concat('wcs'),
            time=astropy.time.Time(np.concatenate([img.time.value for img in self]), format='unix'),
            time_index=concat('time_index'),
            channel=concat('channel'),
            exposure_length=concat('exposure_length'),
            time_wcs=concat('time_wcs'),
        )