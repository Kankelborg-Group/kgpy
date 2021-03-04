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
import kgpy.obs


@dataclasses.dataclass
class HMI(kgpy.obs.Image):
    """
    A class for downloading and storing a sequence of HMI images
    """

    @classmethod
    def from_path_array(
            cls,
            path_array: np.ndarray,
    ) -> 'HMI':

        hmi_map = sunpy.map.Map(path_array[0])
        self = cls.zeros(path_array.shape + (1,) + hmi_map.data.shape)

        for i in range(path_array.shape[0]):
            hmi_map = sunpy.map.Map(path_array[i])
            self.intensity[i,0] = hmi_map.data * u.adu
            self.time[i,0] = hmi_map.date
            self.exposure_length[i,0] = hmi_map.exposure_time
            self.wcs[i,0] = hmi_map.wcs

        return self

    @classmethod
    def from_time_range(
            cls,
            time_start: astropy.time.Time,
            time_end: astropy.time.Time,
            download_path: pathlib.Path = None,
            channels: typ.Optional[u.Quantity] = None,
            user_email: str = 'jacobdparker@gmail.com',
    ):


        if download_path is None:
            download_path = pathlib.Path(__file__).parent / 'data'

        if not download_path.is_dir():
            download_path.mkdir()

        level_1_path = download_path / 'level_1'

        # Initialize JSOC attributes
        time = sunpy.net.attrs.Time(time_start, time_end)
        notify = sunpy.net.attrs.jsoc.Notify(user_email)
        segment = sunpy.net.attrs.jsoc.Segment('image')

        # Download shortwave HMI data
        hmi_series = sunpy.net.attrs.jsoc.Series('hmi.M_45s')


        search = sunpy.net.Fido.search(time, hmi_series, notify)
        files = sunpy.net.Fido.fetch(search, path=str(level_1_path), max_conn=1, progress=False)
        while len(files.errors) > 0:
            files = sunpy.net.Fido.fetch(files, path=str(level_1_path), max_conn=1, progress=False)

        files = sorted(files)

        ####  My current understanding is that no extra calibration is required.
        # files_15 = []
        # for file in files:
        #     file_15 = level_15_path / pathlib.Path(file).name
        #     files_15.append(file_15)
        #     if not file_15.is_file():
        #         hmi_map = sunpy.map.Map(file)
        #         # hmi_map = aiapy.calibrate.register(hmi_map)
        #         hmi_map.save(file_15)



        return cls.from_path_array(np.array(files))
