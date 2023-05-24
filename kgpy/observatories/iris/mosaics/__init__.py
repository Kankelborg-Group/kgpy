from typing import Type
from typing_extensions import Self
import typing as typ
import pathlib
import numpy as np
import numpy.typing as npt
import astropy.units as u
import astropy.time
import astropy.wcs
import astropy.io.fits
from kgpy import obs, img
import kgpy.mixin
import kgpy.labeled
import kgpy.vectors
import kgpy.optics
import kgpy.solar
from . import saa
from . import url

__all__ = ['Cube', 'saa', 'url']


class SpectralRadiance(
    kgpy.solar.SpectralRadiance[
        kgpy.optics.vectors.TemporalOffsetSpectralFieldVector[
            kgpy.labeled.Array,
            kgpy.labeled.Array,
            kgpy.labeled.WorldCoordinateSpace,
            kgpy.labeled.WorldCoordinateSpace,
            kgpy.labeled.WorldCoordinateSpace,
        ],
        kgpy.labeled.Array,
    ],
    kgpy.mixin.Pickleable,
):

    @classmethod
    def from_path_array(
            cls: Type[Self],
            path_array: kgpy.labeled.Array[npt.NDArray[pathlib.Path]],
    ) -> Self:

        shape_base = path_array.shape

        hdu_prototype = astropy.io.fits.open(
            name=str(path_array[dict(time=0, wavelength_base=0)].array),
        )[0]

        wcs_prototype = astropy.wcs.WCS(hdu_prototype)

        shape_wcs = {k:v for k, v in zip(reversed(wcs_prototype.axis_type_names), wcs_prototype.array_shape)}
        shape_wcs = {'wavelength_offset' if k == 'Wavelength' else k:v for k,v in shape_wcs.items()}
        shape_wcs = {'solar_x' if k == 'Solar X' else k:v for k,v in shape_wcs.items()}
        shape_wcs = {'solar_y' if k == 'Solar Y' else k:v for k,v in shape_wcs.items()}
        # shape_wcs['wavelength_offset'] = shape_wcs.pop('Wavelength')
        # shape_wcs['solar_x'] = shape_wcs.pop('Solar X')
        # shape_wcs['solar_y'] = shape_wcs.pop('Solar Y')

        shape = dict(**shape_base, **shape_wcs)

        self = cls(
            input=kgpy.optics.vectors.TemporalOffsetSpectralFieldVector(
                time=kgpy.labeled.Array(
                    array=astropy.time.Time(np.zeros(tuple(shape_base.values())), format='jd'),
                    axes=list(shape_base.keys()),
                ),
                wavelength_base=kgpy.labeled.Array([hdu_prototype.header['LAMREF']] * u.AA, axes=['wavelength_base']),
                wavelength_offset=kgpy.labeled.WorldCoordinateSpace(
                    crval=kgpy.labeled.Array.zeros(shape_base) * u.AA,
                    crpix=kgpy.vectors.CartesianND.from_components(
                        wavelength_offset=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                        solar_x=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                        solar_y=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                    ),
                    cdelt=kgpy.labeled.Array.zeros(shape_base) * (u.AA / u.pix),
                    pc_row=kgpy.vectors.CartesianND.from_components(
                        wavelength_offset=kgpy.labeled.Array.zeros(shape_base),
                        solar_x=kgpy.labeled.Array.zeros(shape_base),
                        solar_y=kgpy.labeled.Array.zeros(shape_base),
                    ),
                    shape_wcs=shape_wcs,
                ),
                field_x=kgpy.labeled.WorldCoordinateSpace(
                    crval=kgpy.labeled.Array.zeros(shape_base) * u.arcsec,
                    crpix=kgpy.vectors.CartesianND.from_components(
                        wavelength_offset=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                        solar_x=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                        solar_y=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                    ),
                    cdelt=kgpy.labeled.Array.zeros(shape_base) * (u.arcsec / u.pix),
                    pc_row=kgpy.vectors.CartesianND.from_components(
                        wavelength_offset=kgpy.labeled.Array.zeros(shape_base),
                        solar_x=kgpy.labeled.Array.zeros(shape_base),
                        solar_y=kgpy.labeled.Array.zeros(shape_base),
                    ),
                    shape_wcs=shape_wcs,
                ),
                field_y=kgpy.labeled.WorldCoordinateSpace(
                    crval=kgpy.labeled.Array.zeros(shape_base) * u.arcsec,
                    crpix=kgpy.vectors.CartesianND.from_components(
                        wavelength_offset=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                        solar_x=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                        solar_y=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                    ),
                    cdelt=kgpy.labeled.Array.zeros(shape_base) * (u.arcsec / u.pix),
                    pc_row=kgpy.vectors.CartesianND.from_components(
                        wavelength_offset=kgpy.labeled.Array.zeros(shape_base),
                        solar_x=kgpy.labeled.Array.zeros(shape_base),
                        solar_y=kgpy.labeled.Array.zeros(shape_base),
                    ),
                    shape_wcs=shape_wcs,
                ),
            ),
            output=kgpy.labeled.Array.zeros(shape) << u.DN,
        )

        i_solar_x = wcs_prototype.axis_type_names.index('Solar X')
        i_solar_y = wcs_prototype.axis_type_names.index('Solar Y')
        i_wavelength_offset = wcs_prototype.axis_type_names.index('Wavelength')

        for index in path_array.ndindex():

            hdu = astropy.io.fits.open(path_array[index].array)[0]

            print('hdu.data.shape', hdu.data.shape)

            wcs = astropy.wcs.WCS(hdu).wcs

            self.output[index] = kgpy.labeled.Array(hdu.data << u.DN, axes=list(shape_wcs.keys()))
            self.input.time[index] = astropy.time.Time(hdu.header['DATE_OBS'])

            self.input.wavelength_offset.crval[index] = (wcs.crval[i_wavelength_offset] << u.AA) - self.input.wavelength_base.broadcast_to(shape_base)[index]
            self.input.wavelength_offset.crpix.wavelength_offset[index] = wcs.crpix[i_wavelength_offset] << u.pix
            self.input.wavelength_offset.crpix.solar_x[index] = wcs.crpix[i_solar_x] << u.pix
            self.input.wavelength_offset.crpix.solar_y[index] = wcs.crpix[i_solar_y] << u.pix
            self.input.wavelength_offset.cdelt[index] = wcs.cdelt[i_wavelength_offset] << (u.AA / u.pix)
            self.input.wavelength_offset.pc_row.wavelength_offset[index] = wcs.pc[i_wavelength_offset, i_wavelength_offset]
            self.input.wavelength_offset.pc_row.solar_x[index] = wcs.pc[i_wavelength_offset, i_solar_x]
            self.input.wavelength_offset.pc_row.solar_y[index] = wcs.pc[i_wavelength_offset, i_solar_y]

            self.input.field_x.crval[index] = wcs.crval[i_solar_x] << u.arcsec
            self.input.field_x.crpix.wavelength_offset[index] = wcs.crpix[i_wavelength_offset] << u.pix
            self.input.field_x.crpix.solar_x[index] = wcs.crpix[i_solar_x] << u.pix
            self.input.field_x.crpix.solar_y[index] = wcs.crpix[i_solar_y] << u.pix
            self.input.field_x.cdelt[index] = wcs.cdelt[i_solar_x] << (u.arcsec / u.pix)
            self.input.field_x.pc_row.wavelength_offset[index] = wcs.pc[i_solar_x, i_wavelength_offset]
            self.input.field_x.pc_row.solar_x[index] = wcs.pc[i_solar_x, i_solar_x]
            self.input.field_x.pc_row.solar_y[index] = wcs.pc[i_solar_x, i_solar_y]

            self.input.field_y.crval[index] = wcs.crval[i_solar_y] << u.arcsec
            self.input.field_y.crpix.wavelength_offset[index] = wcs.crpix[i_wavelength_offset] << u.pix
            self.input.field_y.crpix.solar_x[index] = wcs.crpix[i_solar_x] << u.pix
            self.input.field_y.crpix.solar_y[index] = wcs.crpix[i_solar_y] << u.pix
            self.input.field_y.cdelt[index] = wcs.cdelt[i_solar_y] << (u.arcsec / u.pix)
            self.input.field_y.pc_row.wavelength_offset[index] = wcs.pc[i_solar_y, i_wavelength_offset]
            self.input.field_y.pc_row.solar_x[index] = wcs.pc[i_solar_y, i_solar_x]
            self.input.field_y.pc_row.solar_y[index] = wcs.pc[i_solar_y, i_solar_y]

        self = cls(
            input=kgpy.optics.vectors.TemporalOffsetSpectralFieldVector(
                time=self.input.time,
                wavelength_base=self.input.wavelength_base,
                wavelength_offset=self.input.wavelength_offset.array_labeled,
                field_x=self.input.field_x.array_labeled,
                field_y=self.input.field_y.array_labeled,
            ),
            output=self.output
        )

        return self



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
        path_array = kgpy.labeled.Array(path_array, axes=['time', 'wavelength_base'])

        path_array = path_array[dict(time=slice(4))]

        cube = SpectralRadiance.from_path_array(path_array)

        # cube = Cube.from_path_array(path_array)

        # for i in range(path_array.shape[0]):
        #     for region_x, region_y in zip(saa.regions_x[i], saa.regions_y[i]):
        #         cube.intensity[i, ..., slice(*region_y), slice(*region_x), :] = 0

        # cube.to_pickle(disk_cache)

    return cube


from .mosaics import *
from . import line_profile_moments
