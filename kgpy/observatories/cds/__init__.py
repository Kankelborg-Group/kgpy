import pathlib
import numpy as np
import astropy.time
import astropy.units as u
import astropy.io.fits
import sunpy.net.attrs
from kgpy import obs

__all__ = ['Slice']



class Slice(obs.spectral.Slice):
    """
    A sequence of slices in a single observing program.
    """

    window_keys = [
        'O_4_554_52',
        'FE_16_335_40',
        'HE_1_584_34',
        'HE_2_303_78',
        'MG_9_368_06',
        'O_5_629_73',
    ]

    @classmethod
    def from_path_array(
            cls,
            path_array: np.ndarray,
    ) -> 'Slice':

        ext_index = 1
        hdu_sample = astropy.io.fits.open(path_array[0])[ext_index]

        channels = [key for key in cls.window_keys if key in hdu_sample.header]

        num_times = path_array.shape[0]
        num_channels = len(channels)

        slice_shape = eval(hdu_sample.header['TDIM1'])
        shape = (num_times, num_channels, ) + slice_shape
        self = cls.zeros(shape)
        self.channel = self.channel.value << u.AA

        print(shape)
        print(self.intensity.shape)

        for i in range(num_times):
            print(path_array[i])
            hdu = astropy.io.fits.open(path_array[i])[ext_index]
            hdu.verify('fix')

            for c in range(1, hdu.header['TFIELDS'] + 1):
                key = 'TTYPE' + str(c)
                if hdu.header[key] == 'BACKGROUND':
                    hdu.header[key] = 'BACKGROUND' + str(c)

            for c, chan in enumerate(channels):
                self.intensity[i, c] = hdu.data[0][chan]

        return self

    @classmethod
    def from_time_range(
            cls,
            time_start: astropy.time.Time,
            time_end: astropy.time.Time,
            download_path: pathlib.Path = None,
    ) -> 'Slice':
        if download_path is None:
            download_path = pathlib.Path(__file__).parent / 'data'

        if not download_path.is_dir():
            download_path.mkdir()

        time = sunpy.net.attrs.Time(time_start, time_end)
        instrument = sunpy.net.attrs.Instrument.cds

        search = sunpy.net.Fido.search(time, instrument)
        files = sunpy.net.Fido.fetch(search, path=download_path / '{file}', max_conn=1)

        files = sorted(files)
        files = np.array([pathlib.Path(f) for f in files])

        return cls.from_path_array(files)
