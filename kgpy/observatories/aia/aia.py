import pathlib
import dataclasses
import numpy as np
import datetime
import sunpy
import sunpy.map
import sunpy.net.attrs
import astropy.units as u
from kgpy.io import fits
from aiapy.calibrate import register, update_pointing



@dataclasses.dataclass
class AIA:
    intensity: np.ndarray
    exposure_start_time: np.ndarray
    exposure_length: np.ndarray
    wcs: np.ndarray
    wave: np.ndarray

    @classmethod
    def from_path(cls, name: str, frame_paths = None, directory: pathlib.Path = None):
        if directory is not None:
            frame_paths = np.array(sorted(directory.glob('*')))
        if frame_paths is None:
            print('Make Sure to Provide a Path')
        hdu = fits.load_hdu(frame_paths, hdu_index=0)
        return cls(
            fits.extract_data(hdu),
            # for some reason these files have a dash instead of an underscore for this keyword.
            # That may be a temporary error
            fits.extract_times(hdu, 'DATE-OBS'),
            fits.extract_header_value(hdu, 'EXPTIME'),
            fits.extract_wcs(hdu),
            fits.extract_header_value(hdu, 'WAVE_STR'),
        )



def fetch_from_time(start: datetime.datetime, end: datetime.datetime, download_path: pathlib.Path,
                    aia_channels=None, hmi = False):

    if aia_channels is None:
        aia_channels = [94 * u.AA, 131 * u.AA, 171 * u.AA, 193 * u.AA, 211 * u.AA, 304 * u.AA,
                        335 * u.AA]

    if not download_path.is_dir():
        download_path.mkdir()

    # Initialize JSOC attributes
    time = sunpy.net.attrs.Time(start, end)
    notify = sunpy.net.attrs.jsoc.Notify('jacobdparker@gmail.com')
    segment = sunpy.net.attrs.jsoc.Segment('image')

    # Download shortwave AIA data
    euv_series = sunpy.net.attrs.jsoc.Series('aia.lev1_euv_12s')
    file_paths = []
    for channel in aia_channels:

        # Create a separate path for each channel
        c_path = download_path / str(int(channel.value))
        if not c_path.is_dir():
            c_path.mkdir()

        # Download the data
        search = sunpy.net.Fido.search(time, euv_series, notify, sunpy.net.attrs.jsoc.Wavelength(channel), segment)
        files = sunpy.net.Fido.fetch(search, path=str(c_path),max_conn=1)
        while len(files.errors) > 0:
            print('Found Errors')
            files = sunpy.net.Fido.fetch(files, path=str(c_path),max_conn=1)

        file_paths += files
        # download_jsoc_series(euv_series, c_path, time, notify, jsoc.Wavelength(channel), segment)

    # Make folder for hmi data
    if hmi is True:
        h_path = download_path / 'hmi'
        if not h_path.is_dir():
            h_path.mkdir()
        # Download HMI data
        hmi_series = sunpy.net.attrs.jsoc.Series('hmi.M_45s')
        hmi_search = sunpy.net.Fido.search(time, hmi_series, notify)
        sunpy.net.Fido.fetch(hmi_search, path=str(h_path))

    return file_paths

def aiaprep_from_paths(files):
    """
    Prep downloaded AIA files to level 1.5.  Designed to take the file_paths returned by fetch_from_time as input
    """
    saved_files= []
    for file in files:
        file = pathlib.Path(file)
        map = sunpy.map.Map(file)

        instrument = 'aia.'
        wave = str(map.fits_header['WAVELNTH'])
        time = map.fits_header['T_OBS']

        file_name = pathlib.Path(instrument+wave+'.'+time+'.lev1.5.fits')
        dir_path = file.parent / 'lev15/'

        if not dir_path.is_dir():
            dir_path.mkdir()

        save_path = dir_path/file_name
        saved_files.append(save_path)
        if not save_path.is_file():
            print('Prepping'+str(file))
            map_15 = update_pointing(map)
            map_15 = register(map_15)
            map_15.save(save_path)

    return saved_files



