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
import kgpy.obs


@dataclasses.dataclass
class AIA(kgpy.obs.Image):
    """
    A class for storing downloading and storing a sequence of AIA images
    """

    @classmethod
    def from_path_array(
            cls,
            path_array: np.ndarray,
    ) -> 'AIA':

        aia_map = sunpy.map.Map(path_array[0, 0])
        self = cls.zeros(path_array.shape + aia_map.data.shape)
        self.channel = np.zeros(path_array.shape[~0]) * u.AA

        for i in range(path_array.shape[0]):
            for c in range(path_array.shape[1]):
                aia_map = sunpy.map.Map(path_array[i, c])
                self.intensity[i, c] = aia_map.data * u.adu
                self.time[i, c] = aia_map.date
                self.exposure_length[i, c] = aia_map.exposure_time
                self.channel[c] = aia_map.wavelength
                self.wcs[i, c] = aia_map.wcs

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
                if not file_15.is_file():
                    aia_map = sunpy.map.Map(file)
                    aia_map = aiapy.calibrate.update_pointing(aia_map)
                    aia_map = aiapy.calibrate.register(aia_map)
                    aia_map.save(file_15)

            file_paths.append(files_15)

        file_paths = np.array(file_paths).T

        return cls.from_path_array(file_paths)
