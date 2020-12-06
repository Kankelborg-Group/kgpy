import pathlib
import dataclasses
import numpy as np
import pickle

from esis.data import data


@dataclasses.dataclass
class EIS:
    intensity: np.ndarray
    exposure_start_time: np.ndarray
    exposure_length: np.ndarray
    wcs: np.ndarray
    wave: np.ndarray

    @classmethod
    def from_path(cls, name: str, directory: pathlib.Path):
        frame_paths = np.array(sorted(directory.glob('*')))
        hdu = data.load_hdu(frame_paths, hdu_index=1)

        return cls(
            data.extract_data(hdu),
            data.extract_times(hdu, 'DATE_OBS'),
            data.extract_header_value(hdu, 'EXPTIME'),
            data.extract_wcs(hdu),
            data.extract_header_value(hdu, 'WAVE_STR'),
        )


    def to_pickle(self, path: pathlib.Path):
        file = open(str(path), 'wb')
        pickle.dump(self, file)
        file.close()

        return

    def from_pickle(path: pathlib.Path) -> 'Level1':
        file = open(str(path), 'rb')
        obs = pickle.load(file)
        file.close()

        return obs

