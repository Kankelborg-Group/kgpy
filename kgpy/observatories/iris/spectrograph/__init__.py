import pathlib
import typing as typ
import dataclasses
import numpy as np
import astropy.time
import astropy.units as u
import astropy.constants
import astropy.wcs
import astropy.io.fits
import kgpy.obs
import kgpy.moment

__all__ = ['Cube']


@dataclasses.dataclass
class Cube(kgpy.obs.spectral.Cube):

    @classmethod
    def zeros(cls, shape: typ.Sequence[int]) -> 'Cube':
        self = super().zeros(shape=shape)
        self.time = astropy.time.Time(np.zeros((shape[self.axis.time], shape[~2])), format='unix')
        return self

    @classmethod
    def from_path_array(
            cls,
            path_array: typ.Sequence[pathlib.Path],
            spectral_window: str = 'Si IV 1394',
    ) -> 'Cube':

        for i, path in enumerate(path_array):

            hdu_list = astropy.io.fits.open(str(path))

            hdu_index = 1
            for h in range(len(hdu_list)):
                try:
                    if hdu_list[0].header['TDESC' + str(h)] == spectral_window:
                        hdu_index = h
                except KeyError:
                    pass

            hdu = hdu_list[hdu_index]

            if i == 0:
                # for j, h in enumerate(hdu_list):
                #     print('hdu', j)
                #     print(repr(h.header))

                base_shape = hdu.data.shape
                self = cls.zeros((len(path_array), 1, ) + base_shape)
                self.channel = self.channel.value << u.AA

            d = hdu.data * u.adu

            # self.intensity[i, c] = np.moveaxis(d, 0, ~0)
            self.intensity[i] = d

            self.time[i] = astropy.time.Time(hdu_list[0].header['STARTOBS']) + hdu_list[~1].data[..., 0] * u.s
            self.exposure_length[i] = float(hdu_list[0].header['EXPTIME']) * u.s
            self.channel[:] = float(hdu_list[0].header['TWAVE' + str(hdu_index)]) * u.AA

            wcs = astropy.wcs.WCS(hdu.header)
            self.wcs[i] = wcs

        self.intensity[self.intensity == -200 * u.adu] = np.nan

        return self

    def window_doppler(self, shift_doppler: u.Quantity = 300 * u.km / u.s) -> 'Cube':
        wcs = self.wcs[0, 0]
        wavl_center = self.channel[0]
        wavl_delta = shift_doppler / astropy.constants.c * wavl_center
        wavl_left = wavl_center - wavl_delta
        wavl_right = wavl_center + wavl_delta

        pix_left = int(wcs.world_to_pixel_values(wavl_left.to(u.m), 0 * u.arcsec, 0 * u.arcsec, )[0])
        pix_right = int(wcs.world_to_pixel_values(wavl_right.to(u.m), 0 * u.arcsec, 0 * u.arcsec, )[0])

        wcs_new = self.wcs.copy()
        for i in range(wcs_new.size):
            index = np.unravel_index(i, wcs_new.shape)
            wcs_new[index] = wcs_new[index][..., pix_left:pix_right]

        other = Cube(
            intensity=self.intensity[..., pix_left:pix_right].copy(),
            intensity_uncertainty=self.intensity_uncertainty[..., pix_left:pix_right].copy(),
            wcs=wcs_new,
            time=self.time.copy(),
            time_index=self.time_index.copy(),
            channel=self.channel.copy(),
            exposure_length=self.exposure_length.copy(),
        )

        return other

    @property
    def colors(self):
        intensity = self.intensity.copy()
        thresh_lower = np.nanpercentile(intensity, 1)
        thresh_upper = np.nanpercentile(intensity, 99)
        intensity[intensity < thresh_lower] = thresh_lower
        intensity[intensity > thresh_upper] = thresh_upper

        intensity_avg = np.nanmean(self.intensity, self.axis.perp_axes(self.axis.w))
        divider_left = int(kgpy.moment.percentile.arg_percentile(intensity_avg, 0.33))
        divider_right = int(kgpy.moment.percentile.arg_percentile(intensity_avg, 0.66))

        red = self.intensity[..., :divider_left].sum(~0)
        green = self.intensity[..., divider_left:divider_right + 1].sum(~0)
        blue = self.intensity[..., divider_right + 1:].sum(~0)

        red = red / red.max()
        green = green / green.max()
        blue = blue / blue.max()

        # red = red / np.nanpercentile(red, 99)
        # green = green / np.nanpercentile(green, 99)
        # blue = blue / np.nanpercentile(blue, 99)


        return np.stack([red, green, blue], axis=~0)
