from __future__ import annotations
import pathlib
import numpy as np
import astropy.units as u
import astropy.time
import kgpy.labeled
import kgpy.vectors
import kgpy.function
import kgpy.optics
from ... import spectrograph

__all__ = [
    'spectral_radiance'
]

directory_download = pathlib.Path(__file__).parent / 'data'

start_relative = kgpy.optics.vectors.SpectralFieldVector(
    wavelength=-2 * u.AA,
    field_x=-150 * u.arcsec,
    field_y=-50 * u.arcsec
)
stop_relative = -start_relative

plate_scale = kgpy.optics.vectors.SpectralFieldVector(
    wavelength=(12.72 * (u.mAA / u.pix)).to(u.AA / u.pix),
    field_x=0.33 * (u.arcsec / u.pix),
    field_y=0.33 * (u.arcsec / u.pix),
)

shape_spectral_field = (stop_relative - start_relative) / plate_scale
shape_spectral_field = shape_spectral_field.coordinates
shape_spectral_field = {axis: int(shape_spectral_field[axis] / u.pix) for axis in shape_spectral_field}


def spectral_radiance(
        time_start: astropy.time.Time | None = None,
        time_stop: astropy.time.Time | None = None,
        spectral_window: str = 'Si IV 1394',
        slice_day: slice | None = None,
        use_filament: bool = False,
        rebin_wavelength: int = 1,
        rebin_x: int = 1,
        rebin_y: int = 1,
        discard_south_atlantic_anomaly: bool = True,
) -> spectrograph.SpectralRadiance:

    # query_hek = 'https://www.lmsal.com/hek/hcr?cmd=search-events3' \
    #             '&outputformat=json' \
    #             '&startTime=2013-07-20T00:00' \
    #             '&stopTime=2022-08-05T00:00' \
    #             '&hasData=true' \
    #             '&hideMostLimbScans=true' \
    #             '&obsDesc=F2' \
    #             '&limit=1000'

    query_hek = spectrograph.SpectralRadiance.query_hek_from_time_range(
        time_start=time_start,
        time_stop=time_stop,
        description='F2',
        limit=1000,
    )

    archive_url_list = spectrograph.SpectralRadiance.archive_url_list_from_query_hek(query_hek)

    invalid_dates = [
        '2019/01/07',
        '2020/01/17',
        '2020/01/18',
        '2020/01/24',
        '2020/01/25',
        '2020/01/31',
        '2020/02/01',
        '2020/02/07',
        '2020/02/08',
        '2022/04/30',
    ]

    archive_url_list = np.array([url for url in archive_url_list if not any(date in str(url) for date in invalid_dates)])

    num_pointings_per_day = 3

    archive_url_array = kgpy.labeled.Array(array=np.array(archive_url_list), axes=['time'])
    archive_url_array = archive_url_array.reshape(
        shape=dict(
            time=int(archive_url_array.shape['time'] // num_pointings_per_day),
            pointing=num_pointings_per_day,
        )
    )

    if slice_day is None:
        slice_day = slice(None)
    archive_url_array = archive_url_array[dict(time=slice_day)]

    num_time = archive_url_array.shape['time']

    for index_time in range(num_time):

        result_time = spectrograph.SpectralRadiance.from_archive_url_array(
            archive_url_array=archive_url_array[dict(time=index_time)],
            directory_download=directory_download,
            spectral_window=spectral_window,
            summing_spatial=2,
            use_filament=use_filament,
        )

        result_time.output[result_time.output < -10 * u.DN] = 0

        print('result_time.sum()', result_time.output.sum(axis=('wavelength_offset', 'detector_x', 'detector_y')))

        if discard_south_atlantic_anomaly:
            if np.any(result_time.in_south_atlantic_anomaly):
                print(f'discarding {result_time.input.time.array.min()} due to South Atlantic anomaly.')

        print('result_time.output', result_time.output.shape)

        result_time.output = result_time.output - result_time.background
        # result_time.output = result_time.background
        # result_time.output = result_time.output - result_time.background_fit
        # result_time.output = result_time.background.broadcast_to(result_time.output.shape)

        print('result_time.sum()', result_time.output.sum(axis=('wavelength_offset', 'detector_x', 'detector_y')))

        offset_x = result_time.input.field_x.crval[dict(pointing=num_pointings_per_day // 2)]
        offset_y = result_time.input.field_y.crval[dict(pointing=num_pointings_per_day // 2)]

        result_time = result_time.interp_linear(
            input_new=kgpy.optics.vectors.TemporalOffsetSpectralFieldVector(
                time=None,
                wavelength_base=None,
                wavelength_offset=kgpy.labeled.LinearSpace(
                    start=start_relative.wavelength,
                    stop=stop_relative.wavelength,
                    num=shape_spectral_field['wavelength'],
                    axis='wavelength_offset',
                ),
                field_x=kgpy.labeled.LinearSpace(
                    start=start_relative.field_x + offset_x,
                    stop=stop_relative.field_x + offset_x,
                    num=shape_spectral_field['field_x'],
                    axis='helioprojective_x',
                ),
                field_y=kgpy.labeled.LinearSpace(
                    start=start_relative.field_y + offset_y,
                    stop=stop_relative.field_y + offset_y,
                    num=shape_spectral_field['field_y'],
                    axis='helioprojective_y',
                )
            ),
            axis=['wavelength_offset', 'detector_x', 'detector_y']
        )

        if index_time == 0:
            shape_result = result_time.output.shape
            shape_result = dict(time=num_time, **shape_result)
            print('shape_result', shape_result)
            shape_base = dict(time=num_time)
            result = spectrograph.SpectralRadiance.zeros(
                shape=shape_result,
                axis_wavelength_offset='wavelength_offset',
                axis_detector_x='helioprojective_x',
                axis_detector_y='helioprojective_y',
            )
            result.input.wavelength_base = result_time.input.wavelength_base
            result.input.wavelength_offset = kgpy.labeled.LinearSpace(
                start=kgpy.labeled.Array.zeros(shape_base) * u.AA,
                stop=kgpy.labeled.Array.zeros(shape_base) * u.AA,
                num=shape_spectral_field['wavelength'],
                axis='wavelength_offset',
            )
            result.input.field_x = kgpy.labeled.LinearSpace(
                start=kgpy.labeled.Array.zeros(shape_base) * u.arcsec,
                stop=kgpy.labeled.Array.zeros(shape_base) * u.arcsec,
                num=shape_spectral_field['field_x'],
                axis='helioprojective_x',
            )
            result.input.field_y = kgpy.labeled.LinearSpace(
                start=kgpy.labeled.Array.zeros(shape_base) * u.arcsec,
                stop=kgpy.labeled.Array.zeros(shape_base) * u.arcsec,
                num=shape_spectral_field['field_y'],
                axis='helioprojective_y',
            )

        t = dict(time=index_time)
        t_output = dict(
            **t,
            wavelength_offset=slice(result_time.output.shape['wavelength_offset']),
            helioprojective_x=slice(result_time.output.shape['helioprojective_x']),
            helioprojective_y=slice(result_time.output.shape['helioprojective_y']),
        )
        item = dict(pointing=0, raster=0)
        result.output[t_output] = result_time.output
        result.input.time[t] = result_time.input.time
        result.input.wavelength_offset.start[t] = result_time.input.wavelength_offset.start
        result.input.wavelength_offset.stop[t] = result_time.input.wavelength_offset.stop
        result.input.field_x.start[t] = result_time.input.field_x.start[item]
        result.input.field_x.stop[t] = result_time.input.field_x.stop[item]
        result.input.field_y.start[t] = result_time.input.field_y.start[item]
        result.input.field_y.stop[t] = result_time.input.field_y.stop[item]

    return result
