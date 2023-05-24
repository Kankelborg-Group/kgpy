import typing as typ
import pathlib
import dataclasses
import numpy as np
import datetime
import astropy.time
import astropy.wcs
import astropy.io.fits
import sunpy
import sunpy.map
import sunpy.net.attrs
import astropy.units as u
import aiapy.calibrate
import kgpy.labeled
import kgpy.vectors
import kgpy.function
import kgpy.optics
import kgpy.obs
import kgpy.solar


@dataclasses.dataclass(eq=False)
class RadiantIntensity(
    kgpy.solar.Radiance[
        kgpy.optics.vectors.TemporalSpectralFieldVector[
            kgpy.labeled.Array,
            kgpy.labeled.Array,
            kgpy.labeled.WorldCoordinateSpace,
            kgpy.labeled.WorldCoordinateSpace,
        ],
        kgpy.labeled.Array,
    ],
):
    @classmethod
    def from_path_array(cls, path_array: kgpy.labeled.Array[pathlib.Path]):

        shape_base = path_array.shape
        # map_prototype = sunpy.map.Map(path_array[dict(time=0, wavelength=0)].array)
        # intensity_prototype = kgpy.labeled.Array(map_prototype.data, axes=['pixel_x', 'pixel_y'])
        shape_wcs = dict(pixel_x=4096, pixel_y=4096)
        shape = dict(**shape_base, **shape_wcs)

        time = kgpy.labeled.Array(astropy.time.Time(np.zeros(tuple(shape_base.values())), format='unix'), axes=path_array.axes)
        self = cls(
            input=kgpy.optics.vectors.TemporalSpectralFieldVector(
                time=time,
                wavelength=kgpy.labeled.Array.empty(dict(wavelength=shape['wavelength'])) * u.AA,
                field_x=kgpy.labeled.WorldCoordinateSpace(
                    crval=kgpy.labeled.Array.zeros(shape_base) * u.arcsec,
                    crpix=kgpy.vectors.CartesianND.from_components(
                        pixel_x=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                        pixel_y=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                    ),
                    cdelt=kgpy.labeled.Array.zeros(shape_base) * (u.arcsec / u.pix),
                    pc_row=kgpy.vectors.CartesianND.from_components(
                        pixel_x=kgpy.labeled.Array.zeros(shape_base),
                        pixel_y=kgpy.labeled.Array.zeros(shape_base),
                    ),
                    shape_wcs=shape_wcs,
                ),
                field_y=kgpy.labeled.WorldCoordinateSpace(
                    crval=kgpy.labeled.Array.zeros(shape_base) * u.arcsec,
                    crpix=kgpy.vectors.CartesianND.from_components(
                        pixel_x=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                        pixel_y=kgpy.labeled.Array.zeros(shape_base) * u.pix,
                    ),
                    cdelt=kgpy.labeled.Array.zeros(shape_base) * (u.arcsec / u.pix),
                    pc_row=kgpy.vectors.CartesianND.from_components(
                        pixel_x=kgpy.labeled.Array.zeros(shape_base),
                        pixel_y=kgpy.labeled.Array.zeros(shape_base),
                    ),
                    shape_wcs=shape_wcs,
                ),
            ),
            output=kgpy.labeled.Array.zeros(shape) * u.DN,
        )

        ix = ~0
        iy = ~1

        axes_spatial = [None, None]
        axes_spatial[ix] = 'pixel_x'
        axes_spatial[iy] = 'pixel_y'

        for index in path_array.ndindex():

            aia_map = sunpy.map.Map(path_array[index].array)

            intensity = kgpy.labeled.Array(aia_map.data << u.DN, axes=axes_spatial)
            index_intensity = dict(
                **index,
                pixel_x=slice(intensity.shape['pixel_x']),
                pixel_y=slice(intensity.shape['pixel_y']),
            )

            self.output[index_intensity] = intensity
            # print(aia_map.date)
            self.input.time[index] = aia_map.date
            self.input.wavelength[dict(wavelength=index['wavelength'])] = aia_map.wavelength

            self.input.field_x.crval[index] = aia_map.wcs.wcs.crval[ix]
            self.input.field_x.crpix.pixel_x[index] = aia_map.wcs.wcs.crpix[ix] << u.pix
            self.input.field_x.crpix.pixel_y[index] = aia_map.wcs.wcs.crpix[iy] << u.pix
            self.input.field_x.cdelt[index] = aia_map.wcs.wcs.cdelt[ix] << (u.deg / u.pix)
            self.input.field_x.pc_row.pixel_x[index] = aia_map.wcs.wcs.pc[ix, ix]
            self.input.field_x.pc_row.pixel_y[index] = aia_map.wcs.wcs.pc[ix, iy]

            self.input.field_y.crval[index] = aia_map.wcs.wcs.crval[iy]
            self.input.field_y.crpix.pixel_x[index] = aia_map.wcs.wcs.crpix[ix] << u.pix
            self.input.field_y.crpix.pixel_y[index] = aia_map.wcs.wcs.crpix[iy] << u.pix
            self.input.field_y.cdelt[index] = aia_map.wcs.wcs.cdelt[iy] << (u.deg / u.pix)
            self.input.field_y.pc_row.pixel_x[index] = aia_map.wcs.wcs.pc[iy, ix]
            self.input.field_y.pc_row.pixel_y[index] = aia_map.wcs.wcs.pc[iy, iy]

        return self

    @classmethod
    def from_time_range(
            cls,
            time_start: astropy.time.Time,
            time_end: astropy.time.Time,
            download_path: pathlib.Path = None,
            channels: typ.Optional[u.Quantity] = None,
            user_email: str = 'roytsmart@gmail.com',
    ):
        if channels is None:
            channels = [94, 131, 171, 193, 211, 304, 335] * u.AA

        if download_path is None:
            download_path = pathlib.Path(__file__).parent / 'data'

        if not download_path.is_dir():
            download_path.mkdir()

        level_1_path = download_path / 'level_1'
        level_15_path = download_path / 'level_15'

        if not level_15_path.is_dir():
            level_15_path.mkdir()

        # Initialize JSOC attributes
        time = sunpy.net.attrs.Time(time_start, time_end)
        notify = sunpy.net.attrs.jsoc.Notify(user_email)
        segment = sunpy.net.attrs.jsoc.Segment('image')

        # Download shortwave AIA data
        euv_series = sunpy.net.attrs.jsoc.Series('aia.lev1_euv_12s')
        file_paths = []
        for channel in channels:

            # Download the data
            search = sunpy.net.Fido.search(time, euv_series, notify, sunpy.net.attrs.jsoc.Wavelength(channel), segment)
            files = sunpy.net.Fido.fetch(search, path=str(level_1_path), max_conn=1, progress=False)
            while len(files.errors) > 0:
                files = sunpy.net.Fido.fetch(files, path=str(level_1_path), max_conn=1, progress=False)

            files = sorted(files)

            files_15 = []
            for file in files:
                file_15 = level_15_path / pathlib.Path(file).name
                files_15.append(file_15)
                # files_15.append(file)
                if not file_15.is_file():
                    aia_map = sunpy.map.Map(file)
                    aia_map = aiapy.calibrate.update_pointing(aia_map)
                    aia_map = aiapy.calibrate.register(aia_map)
                    aia_map.save(file_15)

            file_paths.append(files_15)

        file_paths = kgpy.labeled.Array(np.array(file_paths).T, axes=['time', 'wavelength'])

        return cls.from_path_array(file_paths)
