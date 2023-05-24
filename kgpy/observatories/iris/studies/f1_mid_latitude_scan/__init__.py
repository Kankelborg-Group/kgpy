from __future__ import annotations
import pathlib
import numpy as np
import kgpy.labeled
from ... import spectrograph

__all__ = [
    'spectral_radiance'
]

directory_download = pathlib.Path(__file__).parent / 'data'


def spectral_radiance(
        spectral_window: str = 'Si IV 1394',
        use_filament: bool = False,
        num_days: int | None = None,
) -> spectrograph.SpectralRadiance:

    query_hek = 'https://www.lmsal.com/hek/hcr?cmd=search-events3' \
                '&outputformat=json' \
                '&startTime=2013-07-20T00:00' \
                '&stopTime=2022-08-05T00:00' \
                '&hasData=true' \
                '&hideMostLimbScans=true' \
                '&obsDesc=F1' \
                '&limit=1000'

    archive_url_list = spectrograph.SpectralRadiance.archive_url_list_from_query_hek(query_hek)

    invalid_dates = [
        '2019/01/07',
        '2022/04/30',
    ]

    archive_url_list = np.array([url for url in archive_url_list if not any(date in str(url) for date in invalid_dates)])

    if num_days is not None:
        archive_url_list = archive_url_list[:6 * num_days]

    archive_url_array = kgpy.labeled.Array(array=np.array(archive_url_list), axes=['time'])

    return spectrograph.SpectralRadiance.from_archive_url_array(
        archive_url_array=archive_url_array,
        directory_download=directory_download,
        spectral_window=spectral_window,
        use_filament=use_filament,
    )
