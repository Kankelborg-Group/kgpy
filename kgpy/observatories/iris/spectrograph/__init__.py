from __future__ import annotations
import typing as typ
from typing import Type, Sequence, TypeVar
from typing_extensions import Self
import numpy.typing as npt
import os
import sys
import pathlib
import urlpath
import dataclasses
import copy
import shutil
import wget
import paramiko
import requests
import multiprocessing
import matplotlib.pyplot as plt
import numpy
import numpy as np
import scipy.ndimage
import matplotlib.cm
import matplotlib.colors
import matplotlib.animation
import astropy.time
import astropy.units as u
import astropy.constants
import astropy.wcs
import astropy.io.fits
import astropy.coordinates
import astropy.modeling
import sunpy.physics.differential_rotation
import sunpy.coordinates.frames
import kgpy.labeled
import kgpy.obs
import kgpy.moment
import kgpy.mixin
import kgpy.img
import kgpy.optics
import kgpy.solar
from .. import net

__all__ = [
    'SpectralRadiance',
    'Cube',
]

SpectralRadianceT = TypeVar('SpectralRadianceT', bound='SpectralRadiance')


@dataclasses.dataclass
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
    ]
):
    in_south_atlantic_anomaly: kgpy.labeled.ArrayLike = None

    @classmethod
    def from_time_range(
            cls: Type[SpectralRadianceT],
            time_start: astropy.time.Time | None = None,
            time_stop: astropy.time.Time | None = None,
            description: str = '',
            obs_id: None | int = None,
            limit: int = 200,
            directory_download: pathlib.Path | None = None,
            spectral_window: str = 'Si IV 1394',
            summing_spatial: int = 1,
    ):

        query_hek = cls.query_hek_from_time_range(
            time_start=time_start,
            time_stop=time_stop,
            description=description,
            obs_id=obs_id,
            limit=limit,
        )

        return cls.from_query_hek(
            query_hek=query_hek,
            directory_download=directory_download,
            spectral_window=spectral_window,
            summing_spatial=summing_spatial,
        )

    @classmethod
    def query_hek_from_time_range(
            cls: Type[SpectralRadianceT],
            time_start: astropy.time.Time | None = None,
            time_stop: astropy.time.Time | None = None,
            description: str = '',
            obs_id: None | int = None,
            limit: int = 200,
    ):
        format_spec = '%Y-%m-%dT%H:%M'

        if time_start is None:
            time_start = astropy.time.Time('2013-07-20T00:00')

        if time_stop is None:
            time_stop = astropy.time.Time.now()

        query_hek = (
            'https://www.lmsal.com/hek/hcr?cmd=search-events3'
            '&outputformat=json'
            f"&startTime={time_start.strftime(format_spec)}"
            f"&stopTime={time_stop.strftime(format_spec)}"
            '&hasData=true'
            '&hideMostLimbScans=true'
            f'&obsDesc={description}'
            f'&limit={limit}'
        )
        if obs_id is not None:
            query_hek += f'&obsId={obs_id}'

        return query_hek

    @classmethod
    def from_query_hek(
            cls: Type[SpectralRadianceT],
            query_hek: str,
            directory_download: pathlib.Path | None = None,
            spectral_window: str = 'Si IV 1394',
            summing_spatial: int = 1,
    ) -> SpectralRadianceT:

        archive_url_list = cls.archive_url_list_from_query_hek(query_hek)

        return cls.from_archive_url_array(
            archive_url_array=kgpy.labeled.Array(np.array(archive_url_list), axes=['time']),
            directory_download=directory_download,
            spectral_window=spectral_window,
            summing_spatial=summing_spatial,
        )

    @classmethod
    def archive_url_list_from_query_hek(
            cls: Type[SpectralRadianceT],
            query_hek: str,
    ) -> list[urlpath.URL]:

        list_urls = net.get_fits(
            query_text=query_hek,
            raster_only=True,
        )
        list_urls = [urlpath.URL(url) for url in list_urls]
        list_urls = sorted(list_urls, key=lambda x: x.name)

        return list_urls

    @classmethod
    def from_archive_url_array(
            cls: Type[SpectralRadianceT],
            archive_url_array: kgpy.labeled.Array[npt.NDArray[urlpath.URL]],
            directory_download: pathlib.Path | None = None,
            spectral_window: str = 'Si IV 1394',
            summing_spatial: int = 1,
            use_filament: bool = False,
    ) -> SpectralRadianceT:

        archive_path_array = cls.archive_path_array_from_archive_url_array(
            archive_url_array=archive_url_array,
            directory_download=directory_download,
            use_filament=use_filament,
        )
        return cls.from_archive_path_array(
            archive_path_array=archive_path_array,
            spectral_window=spectral_window,
            summing_spatial=summing_spatial,
        )

    @classmethod
    def archive_path_array_from_archive_url_array(
            cls: Type[SpectralRadianceT],
            archive_url_array: kgpy.labeled.Array[npt.NDArray[urlpath.URL]],
            directory_download: pathlib.Path | None = None,
            use_filament: bool = False,
    ):
        if directory_download is None:
            directory_download = pathlib.Path(__file__).parent / 'data'

        archive_path_array = kgpy.labeled.Array(
            array=np.empty(shape=tuple(archive_url_array.shape.values()), dtype=pathlib.Path),
            axes=list(archive_url_array.shape.keys()),
        )

        if use_filament:
            transport = paramiko.Transport(('filament.physics.montana.edu', 22))
            transport.connect(
                # username=os.environ['USERNAME_FILAMENT'],
                username='cbunn',
                # password=os.environ['PASSWORD_FILAMENT'],
                password='cappy8191',
            )
            sftp = paramiko.SFTPClient.from_transport(transport)

        for index in archive_url_array.ndindex():
            archive_url = archive_url_array[index].array
            archive_path = directory_download / archive_url.name
            archive_path_array[index] = archive_path

            if not archive_path.exists():
                directory_extract = cls.folder_from_archive(archive_path)
                if not directory_extract.exists():
                    if not directory_download.exists():
                        directory_download.mkdir()
                    print('downloading', archive_url)
                    if not use_filament:
                        wget.download(url=str(archive_url), out=str(archive_path), bar=wget.bar_adaptive)
                    else:
                        archive_path_filament = pathlib.Path(r'/exports/fi1/IRIS/archive/level2')
                        archive_path_filament = archive_path_filament / archive_url.relative_to(archive_url.parents[4])
                        sftp.get(archive_path_filament.as_posix(), archive_path)

        if use_filament:
            sftp.close()
            transport.close()

        return archive_path_array

    @classmethod
    def from_archive_path_array(
            cls: Type[SpectralRadianceT],
            archive_path_array: kgpy.labeled.Array[npt.NDArray[pathlib.Path]],
            spectral_window: str = 'Si IV 1394',
            summing_spatial: int = 1,
    ) -> SpectralRadianceT:

        fits_path_array = cls.fits_path_array_from_archive_path_array(archive_path_array)
        return cls.from_fits_path_array(
            fits_path_array=fits_path_array,
            spectral_window=spectral_window,
            summing_spatial=summing_spatial,
        )

    @classmethod
    def fits_path_array_from_archive_path_array(
            cls: Type[SpectralRadianceT],
            archive_path_array: kgpy.labeled.Array[npt.NDArray[pathlib.Path]],
    ) -> kgpy.labeled.Array[npt.NDArray[pathlib.Path]]:
        # fits_path_list = []
        i = 0
        for index in archive_path_array.ndindex():
            archive = archive_path_array[index].array
            extract_dir = cls.folder_from_archive(archive)
            if not extract_dir.exists():
                shutil.unpack_archive(filename=archive, extract_dir=extract_dir)
            # if archive.exists():
            #     archive.unlink()
            fits_path_list = sorted(extract_dir.rglob('*.fits'))
            num_rasters_per_archive = len(fits_path_list)
            if i == 0:
                shape = dict(**archive_path_array.shape, raster=num_rasters_per_archive)
                fits_path_array = kgpy.labeled.Array(
                    array=np.empty(tuple(shape.values()), dtype=pathlib.Path),
                    axes=list(shape.keys())
                )
            fits_path_array[index] = kgpy.labeled.Array(np.array(fits_path_list), axes=['raster'])
            i = i + 1

        # fits_path_array = kgpy.labeled.Array(np.array(fits_path_list), axes=['time'])

        return fits_path_array

    @classmethod
    def folder_from_archive(cls: Type[Self], archive: pathlib.Path) -> pathlib.Path:
        return archive.parent / pathlib.Path(archive.stem).stem

    @classmethod
    def from_fits_path_array(
            cls: Type[SpectralRadianceT],
            fits_path_array: kgpy.labeled.Array[npt.NDArray[pathlib.Path]],
            spectral_window: str = 'Si IV 1394',
            summing_spatial: int = 1,
    ) -> SpectralRadianceT:
        hdu_list_prototype = astropy.io.fits.open(str(fits_path_array.reshape(dict(dummy=-1))[dict(dummy=0)].array))

        hdu_index = 1
        for h in range(len(hdu_list_prototype)):
            try:
                if hdu_list_prototype[0].header['TDESC' + str(h)] == spectral_window:
                    hdu_index = h
            except KeyError:
                pass

        hdu_prototype = hdu_list_prototype[hdu_index]
        wcs_prototype = astropy.wcs.WCS(hdu_prototype)

        pixels_extra = 2

        shape_wcs = {k:v + pixels_extra for k, v in zip(reversed(wcs_prototype.axis_type_names), wcs_prototype.array_shape)}
        shape_wcs = {'wavelength_offset' if k == 'WAVE' else k:v for k,v in shape_wcs.items()}
        shape_wcs = {'detector_x' if k == 'HPLN' else k:v for k,v in shape_wcs.items()}
        shape_wcs = {'detector_y' if k == 'HPLT' else k:v for k,v in shape_wcs.items()}

        summing_spatial_current = hdu_list_prototype[0].header['SUMSPAT']
        ratio_summing_spatial = summing_spatial // summing_spatial_current

        if ratio_summing_spatial > 1 :
            shape_wcs['detector_y'] = shape_wcs['detector_y'] // ratio_summing_spatial
        elif ratio_summing_spatial < 1:
            raise ValueError(
                f'Current spatial summing ({summing_spatial_current}) is already larger than target spatial summing '
                f'({summing_spatial})'
            )

        shape_base = fits_path_array.shape
        shape = {**shape_base, **shape_wcs}

        self = cls.zeros(
            shape=shape,
            axis_wavelength_offset='wavelength_offset',
            axis_detector_x='detector_x',
            axis_detector_y='detector_y',
        )

        self.input.wavelength_base = hdu_list_prototype[0].header[f'TWAVE{hdu_index}'] * u.AA

        i_solar_x = wcs_prototype.axis_type_names.index('HPLN')
        i_solar_y = wcs_prototype.axis_type_names.index('HPLT')
        i_wavelength_offset = wcs_prototype.axis_type_names.index('WAVE')

        for index in fits_path_array.ndindex():

            fits_path = fits_path_array[index].array

            print(fits_path)

            hdu_list = astropy.io.fits.open(fits_path)
            hdu = hdu_list[hdu_index]

            print('hdu.data.shape', hdu.data.shape)

            wcs = astropy.wcs.WCS(hdu).wcs

            output = kgpy.labeled.Array(hdu.data << u.DN, axes=list(shape_wcs.keys()))

            summing_spatial_current = hdu_list[0].header['SUMSPAT']
            ratio_summing_spatial = summing_spatial // summing_spatial_current

            output = output[dict(detector_y=slice(output.shape['detector_y'] // ratio_summing_spatial * ratio_summing_spatial))]

            sh = dict()
            for axis in output.shape:
                if axis == 'detector_y':
                    sh[axis] = output.shape[axis] // ratio_summing_spatial
                    sh['rebin_y'] = ratio_summing_spatial
                else:
                    sh[axis] = output.shape[axis]

            output = output.reshape(sh).sum('rebin_y')

            index_output = dict(
                **index,
                wavelength_offset=slice(output.shape['wavelength_offset']),
                detector_x=slice(output.shape['detector_x']),
                detector_y=slice(output.shape['detector_y']),
            )
            self.output[index_output] = output
            self.input.time[index] = astropy.time.Time(hdu_list[0].header['DATE_OBS'])

            self.input.wavelength_offset.crval[index] = (wcs.crval[i_wavelength_offset] << u.m).to(u.AA) - self.input.wavelength_base
            self.input.wavelength_offset.crpix.coordinates['wavelength_offset'][index] = wcs.crpix[i_wavelength_offset] << u.pix
            self.input.wavelength_offset.crpix.coordinates['detector_x'][index] = wcs.crpix[i_solar_x] << u.pix
            self.input.wavelength_offset.crpix.coordinates['detector_y'][index] = wcs.crpix[i_solar_y] << u.pix
            self.input.wavelength_offset.cdelt[index] = (wcs.cdelt[i_wavelength_offset] << (u.m / u.pix)).to(u.AA / u.pix)
            self.input.wavelength_offset.pc_row.coordinates['wavelength_offset'][index] = wcs.pc[i_wavelength_offset, i_wavelength_offset]
            self.input.wavelength_offset.pc_row.coordinates['detector_x'][index] = wcs.pc[i_wavelength_offset, i_solar_x]
            self.input.wavelength_offset.pc_row.coordinates['detector_y'][index] = wcs.pc[i_wavelength_offset, i_solar_y]

            self.input.field_x.crval[index] = wcs.crval[i_solar_x] << u.deg
            self.input.field_x.crpix.coordinates['wavelength_offset'][index] = wcs.crpix[i_wavelength_offset] << u.pix
            self.input.field_x.crpix.coordinates['detector_x'][index] = wcs.crpix[i_solar_x] << u.pix
            self.input.field_x.crpix.coordinates['detector_y'][index] = wcs.crpix[i_solar_y] << u.pix
            self.input.field_x.cdelt[index] = wcs.cdelt[i_solar_x] << (u.deg / u.pix)
            self.input.field_x.pc_row.coordinates['wavelength_offset'][index] = wcs.pc[i_solar_x, i_wavelength_offset]
            self.input.field_x.pc_row.coordinates['detector_x'][index] = wcs.pc[i_solar_x, i_solar_x]
            self.input.field_x.pc_row.coordinates['detector_y'][index] = wcs.pc[i_solar_x, i_solar_y]

            self.input.field_y.crval[index] = wcs.crval[i_solar_y] << u.deg
            self.input.field_y.crpix.coordinates['wavelength_offset'][index] = wcs.crpix[i_wavelength_offset] << u.pix
            self.input.field_y.crpix.coordinates['detector_x'][index] = wcs.crpix[i_solar_x] << u.pix
            self.input.field_y.crpix.coordinates['detector_y'][index] = (wcs.crpix[i_solar_y] << u.pix) / ratio_summing_spatial
            self.input.field_y.cdelt[index] = (wcs.cdelt[i_solar_y] << (u.deg / u.pix)) * ratio_summing_spatial
            self.input.field_y.pc_row.coordinates['wavelength_offset'][index] = wcs.pc[i_solar_y, i_wavelength_offset]
            self.input.field_y.pc_row.coordinates['detector_x'][index] = wcs.pc[i_solar_y, i_solar_x]
            self.input.field_y.pc_row.coordinates['detector_y'][index] = wcs.pc[i_solar_y, i_solar_y]

            print('SAA', hdu_list[0].header['SAA'])
            print('HLZ', hdu_list[0].header['HLZ'])
            print('NSPIKES', hdu_list[0].header['NSPIKES'])
            print(f'TDRMS{hdu_index}', hdu_list[0].header[f'TDRMS{hdu_index}'])
            print(f'TDP99_{hdu_index}', hdu_list[0].header[f'TDP99_{hdu_index}'])
            self.in_south_atlantic_anomaly[index] = hdu_list[0].header['SAA'] == 'IN'

        # self = cls(
        #     input=kgpy.optics.vectors.TemporalOffsetSpectralFieldVector(
        #         time=self.input.time,
        #         wavelength_base=self.input.wavelength_base,
        #         wavelength_offset=self.input.wavelength_offset.array_labeled,
        #         field_x=self.input.field_x.array_labeled,
        #         field_y=self.input.field_y.array_labeled,
        #     ),
        #     output=self.output
        # )

        return self

    @classmethod
    def zeros(
            cls: Type[Self],
            shape: dict[str, int],
            axis_wavelength_offset: str = 'wavelength_offset',
            axis_detector_x: str = 'detector_x',
            axis_detector_y: str = 'detector_y',
    ) -> Self:

        shape_base = shape.copy()
        shape_base.pop(axis_wavelength_offset)
        shape_base.pop(axis_detector_x)
        shape_base.pop(axis_detector_y)

        shape_wcs = {k: shape[k] for k in shape if k not in shape_base}

        return cls(
            input=kgpy.optics.vectors.TemporalOffsetSpectralFieldVector(
                time=kgpy.labeled.Array(
                    array=astropy.time.Time(np.zeros(tuple(shape_base.values())), format='jd'),
                    axes=list(shape_base.keys()),
                ),
                wavelength_base=kgpy.labeled.Array([0] * u.AA, axes=['wavelength_base']),
                wavelength_offset=kgpy.labeled.WorldCoordinateSpace(
                    crval=kgpy.labeled.Array.zeros(shape_base) * u.AA,
                    crpix=kgpy.vectors.CartesianND.from_components(
                        wavelength_offset=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                        detector_x=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                        detector_y=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                    ),
                    cdelt=kgpy.labeled.Array.zeros(shape_base) * (u.AA / u.pix),
                    pc_row=kgpy.vectors.CartesianND.from_components(
                        wavelength_offset=kgpy.labeled.Array.zeros(shape_base),
                        detector_x=kgpy.labeled.Array.zeros(shape_base),
                        detector_y=kgpy.labeled.Array.zeros(shape_base),
                    ),
                    shape_wcs=shape_wcs,
                ),
                field_x=kgpy.labeled.WorldCoordinateSpace(
                    crval=kgpy.labeled.Array.zeros(shape_base) * u.arcsec,
                    crpix=kgpy.vectors.CartesianND.from_components(
                        wavelength_offset=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                        detector_x=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                        detector_y=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                    ),
                    cdelt=kgpy.labeled.Array.zeros(shape_base) * (u.arcsec / u.pix),
                    pc_row=kgpy.vectors.CartesianND.from_components(
                        wavelength_offset=kgpy.labeled.Array.zeros(shape_base),
                        detector_x=kgpy.labeled.Array.zeros(shape_base),
                        detector_y=kgpy.labeled.Array.zeros(shape_base),
                    ),
                    shape_wcs=shape_wcs,
                ),
                field_y=kgpy.labeled.WorldCoordinateSpace(
                    crval=kgpy.labeled.Array.zeros(shape_base) * u.arcsec,
                    crpix=kgpy.vectors.CartesianND.from_components(
                        wavelength_offset=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                        detector_x=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                        detector_y=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                    ),
                    cdelt=kgpy.labeled.Array.zeros(shape_base) * (u.arcsec / u.pix),
                    pc_row=kgpy.vectors.CartesianND.from_components(
                        wavelength_offset=kgpy.labeled.Array.zeros(shape_base),
                        detector_x=kgpy.labeled.Array.zeros(shape_base),
                        detector_y=kgpy.labeled.Array.zeros(shape_base),
                    ),
                    shape_wcs=shape_wcs,
                ),
            ),
            output=kgpy.labeled.Array.zeros(shape) << u.DN,
            in_south_atlantic_anomaly=kgpy.labeled.Array.zeros(shape_base, dtype=bool),
        )

    @property
    def background(self: SpectralRadianceT) -> kgpy.labeled.Array:
        axis = self.output.axes.copy()
        axis.remove('wavelength_offset')
        axis.remove('detector_x')
        axis.remove('detector_y')
        # result = np.median(self.output, axis=axis)
        result = self.output
        result = result.filter_mean_trimmed(shape_kernel=dict(detector_x=11, ))
        result = result.filter_mean_trimmed(shape_kernel=dict(detector_y=11, ))
        result = result.filter_mean_trimmed(shape_kernel=dict(wavelength_offset=51,), proportion=0.35)
        # result = result.filter_mean(shape_kernel=dict(wavelength_offset=11, detector_x=11, detector_y=11))

        # spectrum = np.median(result, axis=('detector_x', 'detector_y'))
        #
        # result_bg = result - spectrum
        # result_bg = np.median(result_bg, axis='detector_x')
        #
        # result = result - result_bg

        return result

    @classmethod
    def _background_fit_kernel(
            cls,
            wavelength_index: kgpy.labeled.AbstractArray,
            output_index: kgpy.labeled.AbstractArray,
    ):
        model_background = astropy.modeling.models.Polynomial1D(degree=2, c0=2 * u.DN)
        model_background.c0.min = -2 * u.DN

        model_si_iv = astropy.modeling.models.Gaussian1D(amplitude=2 * u.DN, mean=0 * u.AA,stddev=0.1 * u.AA)
        model_si_iv.amplitude.min = 0 * u.DN
        # model_si_iv.mean.fixed = True
        model_si_iv.mean.min = -0.2 * u.AA
        model_si_iv.mean.max = 0.2 * u.AA
        # model_si_iv.stddev.max = 0.5 * u.AA

        model = model_background + model_si_iv

        # fitter = astropy.modeling.fitting.LevMarLSQFitter()
        fitter = astropy.modeling.fitting.SimplexLSQFitter()

        x = wavelength_index.array
        model = fitter(
            model=model,
            x=x,
            y=output_index.array,
        )

        model_si_iv = model[1]

        # print(model)

        # return kgpy.labeled.Array(model(x), axes=output_index.axes)

        output_si_iv = kgpy.labeled.Array(model_si_iv(x), axes=output_index.axes)

        return output_index - output_si_iv

    @property
    def background_fit(self: SpectralRadianceT) -> kgpy.labeled.Array:

        shape = self.shape

        result = self.output.copy()

        output = self.output.broadcast_to(shape)
        wavelength = self.input.wavelength_offset.broadcast_to(shape)
        field_y = self.input.field_y.broadcast_to(shape)

        print('self.output', self.output.shape)
        print('wavelength', wavelength.shape)
        print('field_y', field_y.shape)

        print('starting multiprocessing pool')

        with multiprocessing.Pool() as pool:
            background = pool.starmap(
                func=self._background_fit_kernel,
                iterable=((wavelength[index], output[index]) for index in kgpy.labeled.ndindex(shape, axis_ignored='wavelength_offset')),
            )

        # background = map(
        #     self._background_fit_kernel,
        #     (wavelength[index] for index in index_iterator),
        #     (output[index] for index in index_iterator),
        # )

        print('end of multiprocessing pool')

        print('result.sum()', result.sum())

        for index, bg in zip(kgpy.labeled.ndindex(shape, axis_ignored='wavelength_offset'), background):
            result[index] = bg

        print('result.sum()', result.sum())

        # for index in kgpy.labeled.ndindex(shape, axis_ignored=('wavelength_offset', 'detector_y')):
        #
        #     print('index', index)
        #
        #     model_background = astropy.modeling.models.Polynomial2D(degree=2)
        #     model_si_iv = astropy.modeling.models.Gaussian2D(x_stddev=0.5, y_stddev=1e6)
        #     model_si_iv.y_stddev.fixed = True
        #     model_si_iv.y_mean.fixed = True
        #
        #     model = model_background + model_si_iv
        #
        #     x = wavelength[index].array.value
        #     y = field_y[index].array.value
        #     model = fitter(
        #         model=model,
        #         x=x,
        #         y=y,
        #         z=output[index].array,
        #     )
        #
        #     print(model)
        #
        #     model_background = model[0]
        #
        #     result[index] = kgpy.labeled.Array(model_background(x, y), axes=self.output[index].axes)

        result = result.filter_median(shape_kernel=dict(detector_x=21, ))

        return result


@dataclasses.dataclass
class Cube(kgpy.obs.spectral.Cube):
    time_wcs: typ.Optional[u.Quantity] = None

    @classmethod
    def zeros(cls, shape: typ.Sequence[int]) -> 'Cube':
        self = super().zeros(shape=shape)
        sh = shape[:-cls.axis.num_right_dim]
        self.time = astropy.time.Time(np.zeros(sh + (shape[cls.axis.y],)), format='unix')
        self.time_wcs = np.zeros(self.time.shape) * u.s
        return self

    @classmethod
    def from_archive(
            cls,
            archive: pathlib.Path,
            spectral_window: str = 'Si IV 1394',
    ) -> 'Cube':
        extract_dir = archive.parent / pathlib.Path(archive.stem).stem
        if not extract_dir.exists():
            shutil.unpack_archive(filename=archive, extract_dir=extract_dir)
        path_sequence = sorted(extract_dir.rglob('*.fits'))
        return cls.from_path_sequence(path_sequence=path_sequence, spectral_window=spectral_window)

    @classmethod
    def from_path_sequence(
            cls,
            path_sequence: typ.Sequence[pathlib.Path],
            spectral_window: str = 'Si IV 1394',
    ) -> 'Cube':

        hdu_list = astropy.io.fits.open(str(path_sequence[0]))
        hdu_index = 1
        for h in range(len(hdu_list)):
            try:
                if hdu_list[0].header['TDESC' + str(h)] == spectral_window:
                    hdu_index = h
            except KeyError:
                pass
        hdu = hdu_list[hdu_index]

        base_shape = hdu.data.shape
        self = cls.zeros((len(path_sequence), 1,) + base_shape)
        self.channel = self.channel.value << u.AA

        for i, path in enumerate(path_sequence):
            hdu_list = astropy.io.fits.open(str(path))
            hdu = hdu_list[hdu_index]

            d = hdu.data * u.adu

            # self.intensity[i, c] = np.moveaxis(d, 0, ~0)
            self.intensity[i] = d

            self.time_wcs[i] = hdu_list[~1].data[..., 0] * u.s
            self.time[i] = astropy.time.Time(hdu_list[0].header['STARTOBS']) + self.time_wcs[i]
            self.exposure_length[i] = float(hdu_list[0].header['EXPTIME']) * u.s
            self.channel[:] = float(hdu_list[0].header['TWAVE' + str(hdu_index)]) * u.AA

            wcs = astropy.wcs.WCS(hdu.header)
            self.wcs[i] = wcs

        self.intensity[self.intensity == -200 * u.adu] = np.nan

        return self

    @property
    def intensity_despiked(self):
        return kgpy.img.spikes.identify_and_fix(
            data=self.intensity.value,
            axis=(~2, ~1, ~0),
            # axis=(0, ~2, ~1, ~0),
            kernel_size=5,
        )[0] << self.intensity.unit

    def window_doppler(self, shift_doppler: u.Quantity = 300 * u.km / u.s) -> 'Cube':

        wavl_center = self.channel[0]
        wavl_delta = shift_doppler / astropy.constants.c * wavl_center
        wavl_left = wavl_center - wavl_delta
        wavl_right = wavl_center + wavl_delta

        wcs_new = self.wcs.copy()
        for i in range(self.wcs.size):
            print('i', i)
            index = np.unravel_index(i, self.wcs.shape)
            pix_left = int(self.wcs[index].world_to_pixel_values(wavl_left.to(u.m), 0 * u.arcsec, 0 * u.arcsec, )[0])
            pix_right = int(self.wcs[index].world_to_pixel_values(wavl_right.to(u.m), 0 * u.arcsec, 0 * u.arcsec, )[0]) + 1
            pix_left = np.maximum(pix_left, 0)
            pix_right = np.minimum(pix_right, self.intensity.shape[~0])
            print('pix_left', pix_left)
            print('pix_right', pix_right)
            if i == 0:
                intensity = np.empty(self.intensity.shape[:~0] + (pix_right - pix_left,))
            print('intensity.shape', intensity.shape)
            print('self.intensity.shape', self.intensity.shape)
            intensity[index] = self.intensity[index][..., pix_left:pix_right]
            wcs_new[index] = wcs_new[index][..., pix_left:pix_right]

        other = Cube(
            intensity=intensity,
            # intensity_uncertainty=self.intensity_uncertainty[..., pix_left:pix_right].copy(),
            wcs=wcs_new,
            time=self.time.copy(),
            time_index=self.time_index.copy(),
            channel=self.channel.copy(),
            exposure_length=self.exposure_length.copy(),
            time_wcs=self.time_wcs.copy()
        )

        return other

    @property
    def colormap_spectrum(self) -> matplotlib.cm.ScalarMappable:
        colormap = matplotlib.cm.get_cmap('gist_rainbow')
        segment_data = colormap._segmentdata.copy()

        last_segment = ~1
        segment_data['red'] = segment_data['red'][:last_segment].copy()
        segment_data['green'] = segment_data['green'][:last_segment].copy()
        segment_data['blue'] = segment_data['blue'][:last_segment].copy()
        segment_data['alpha'] = segment_data['alpha'][:last_segment].copy()

        segment_data['red'][:, 0] /= segment_data['red'][~0, 0]
        segment_data['green'][:, 0] /= segment_data['green'][~0, 0]
        segment_data['blue'][:, 0] /= segment_data['blue'][~0, 0]
        segment_data['alpha'][:, 0] /= segment_data['alpha'][~0, 0]

        colormap = matplotlib.colors.LinearSegmentedColormap(
            name='spectrum',
            segmentdata=segment_data,
        )
        mappable = matplotlib.cm.ScalarMappable(
            cmap=colormap.reversed(),
        )
        return mappable

    def colors(self, shift_doppler_max: u.Quantity = 50 * u.km / u.s):

        intensity = np.nan_to_num(self.intensity)

        wcs = self.wcs[0, 0]
        wavl_center = self.channel[0]
        shift_doppler = 50 * u.km / u.s
        wavl_delta = shift_doppler / astropy.constants.c * wavl_center
        wavl_left = wavl_center - wavl_delta
        wavl_right = wavl_center + wavl_delta
        pix_mask_left = int(wcs.world_to_pixel_values(wavl_left.to(u.m), 0 * u.arcsec, 0 * u.arcsec, )[0])
        pix_mask_right = int(wcs.world_to_pixel_values(wavl_right.to(u.m), 0 * u.arcsec, 0 * u.arcsec, )[0]) + 1

        intensity_background = np.median(intensity, axis=2, keepdims=True)
        intensity_background[..., pix_mask_left:pix_mask_right] = np.median(intensity_background, axis=~0, keepdims=True)
        intensity = intensity - intensity_background
        del intensity_background

        intensity_max = np.percentile(intensity, 99, self.axis.perp_axes(self.axis.w))
        intensity_min = 0
        intensity = (intensity - intensity_min) / (intensity_max - intensity_min)
        del intensity_max

        mappable = self.colormap_spectrum

        # shift_doppler = 50 * u.km / u.s

        wavl_delta = shift_doppler_max / astropy.constants.c * wavl_center
        wavl_left = wavl_center - wavl_delta
        wavl_right = wavl_center + wavl_delta
        pix_left = int(wcs.world_to_pixel_values(wavl_left.to(u.m), 0 * u.arcsec, 0 * u.arcsec, )[0])
        pix_right = int(wcs.world_to_pixel_values(wavl_right.to(u.m), 0 * u.arcsec, 0 * u.arcsec, )[0]) + 1
        index = np.expand_dims(np.arange(pix_left, pix_right), axis=self.axis.perp_axes(self.axis.w))
        color = np.sum(mappable.to_rgba(index) * intensity[..., pix_left:pix_right, np.newaxis], axis=~1)
        color = color / np.sum(mappable.to_rgba(index), axis=~1)

        color_max = np.max(color[..., :~0], axis=~0)
        # color[..., ~0] = color[..., :~0].sum(~0) / 3
        color[..., ~0] = np.nan_to_num(np.sqrt(color_max))

        # color[..., ~0] = np.sqrt(color[..., ~0]).real
        # color[... :~0] = color[..., :~0] / np.nanmax(color[..., :~0], axis=~0, keepdims=True)

        mask = color_max > 0
        color[mask, :~0] = color[mask, :~0] / color_max[mask, np.newaxis]

        return color

    def animate_colors(
            self,
            ax: matplotlib.axes.Axes,
            channel_index: int = 0,
            thresh_min: u.Quantity = 0.01 * u.percent,
            thresh_max: u.Quantity = 99.9 * u.percent,
            frame_interval: u.Quantity = 1 * u.s,
            max_doppler_shift: u.Quantity = 50 * u.km / u.s,
            align_frames: bool = True,
            repeat_factor: int = 1
    ) -> matplotlib.animation.FuncAnimation:

        other = self.window_doppler(shift_doppler=300 * u.km / u.s)
        data = other.colors(max_doppler_shift)[:, channel_index]

        if align_frames:

            pad = 300
            pads = [pad, pad]
            data = np.swapaxes(data, 1, 2)
            data = np.pad(data, pad_width=[[0, 0], pads, pads, [0, 0]])

            index_reference = 0
            slice_reference = index_reference, channel_index

            reference_coordinate = astropy.coordinates.SkyCoord(
                Tx=self.wcs[slice_reference].wcs.crval[2] * u.deg,
                Ty=self.wcs[slice_reference].wcs.crval[1] * u.deg,
                obstime=self.time[slice_reference][0],
                observer="earth",
                frame=sunpy.coordinates.frames.Helioprojective
            )

            shift_x_min, shift_y_min = 0, 0
            shift_x_max, shift_y_max = 0, 0

            for i in range(data.shape[0]):
                if i == index_reference:
                    continue

                reference_coordinate_rotated = sunpy.physics.differential_rotation.solar_rotate_coordinate(
                    coordinate=reference_coordinate,
                    time=self.time[i, channel_index][0],
                )

                shift_x = reference_coordinate_rotated.Tx - self.wcs[i, channel_index].wcs.crval[2] * u.deg
                shift_y = reference_coordinate_rotated.Ty - self.wcs[i, channel_index].wcs.crval[1] * u.deg

                if np.isfinite(shift_x) and np.isfinite(shift_y):

                    shift_x = -int(shift_x / (self.wcs[i, channel_index].wcs.cdelt[2] * u.deg))
                    shift_y = -int(shift_y / (self.wcs[i, channel_index].wcs.cdelt[1] * u.deg))
                    shift = np.array([shift_y, shift_x, 0])
                    print(shift)

                    data_shifted = np.fft.ifftn(scipy.ndimage.fourier_shift(np.fft.fftn(data[i]), shift)).real
                    data[i] = data_shifted
                    self.wcs[i, channel_index].wcs.crpix[2] += shift_x
                    self.wcs[i, channel_index].wcs.crpix[1] += shift_y
                    # mask_shifted = np.isclose(data_shifted, 0)
                    # data[i, mask_shifted] = data[i - 1, mask_shifted]

                else:
                    shift_x = 0 * u.deg
                    shift_y = 0 * u.deg

                shift_x_min = min(shift_x_min, shift_x)
                shift_y_min = min(shift_y_min, shift_y)
                shift_x_max = max(shift_x_max, shift_x)
                shift_y_max = max(shift_y_max, shift_y)

            print('shift_min', shift_x_min, shift_y_min)
            print('shift_max', shift_x_max, shift_y_max)

            slice_y = slice(pad + int(shift_y_min), -pad + int(shift_y_max))
            slice_x = slice(pad + int(shift_x_min), -pad + int(shift_x_max))
            data = data[..., slice_y, slice_x, :]
            data = np.swapaxes(data, 1, 2)
        data = np.clip(data, 0, 1)
        #
        # data[..., ~0] = .1
        #

        def func(i: int):

            i = i // repeat_factor

            ax.clear()

            image = data[i]
            ix, iy = np.indices((sz + 1 for sz in image.shape[:~0]))
            wcs = self.wcs[i, channel_index].dropaxis(0)
            # wcs = wcs[slice_y, slice_x]

            coords = wcs.array_index_to_world(ix, iy)
            x = coords.Tx
            y = coords.Ty

            ax.pcolormesh(
                x.value,
                y.value,
                image,
            )

            ax.text(
                x=0,
                y=0,
                s=astropy.time.Time(np.mean(self.time[i, channel_index].to_value('unix')), format='unix').strftime('%Y-%m-%d %H:%M:%S'),
                ha='left',
                va='bottom',
                color='black',
                transform=plt.gcf().transFigure,
            )

        # img = ax.imshow(
        #     X=data[0],
        #     origin='lower',
        # )
        #
        # text = ax.text(
        #     x=0,
        #     y=0,
        #     s=astropy.time.Time(np.mean(self.time[0, channel_index].to_value('unix')), format='unix').strftime('%Y-%m-%d %H:%M:%S'),
        #     ha='left',
        #     va='bottom',
        #     color='white',
        # )
        #
        # def func(i: int):
        #     img.set_data(data[i])
        #     text.set_text(astropy.time.Time(np.mean(self.time[i, channel_index].to_value('unix')), format='unix').strftime('%Y-%m-%d %H:%M:%S'))

        return matplotlib.animation.FuncAnimation(
            fig=ax.figure,
            func=func,
            # frames=20,
            frames=data.shape[0] * repeat_factor,
            interval=frame_interval.to(u.ms).value / repeat_factor,
        )


@dataclasses.dataclass
class CubeList(kgpy.mixin.DataclassList[Cube]):

    def to_cube(self) -> Cube:
        def concat(param: str):
            return np.concatenate([getattr(cube, param) for cube in self])

        return Cube(
            intensity=concat('intensity'),
            # intensity_uncertainty=concat('intensity_uncertainty'),
            wcs=concat('wcs'),
            time=astropy.time.Time(np.concatenate([img.time.value for img in self]), format='unix'),
            time_index=concat('time_index'),
            channel=concat('channel'),
            exposure_length=concat('exposure_length'),
            time_wcs=concat('time_wcs'),
        )
