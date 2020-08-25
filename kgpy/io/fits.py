import astropy.wcs
import astropy.io.fits
import astropy.time
import pathlib
import numpy as np


def load_hdu(frame_path: np.ndarray, hdu_index: int = 0) -> np.ndarray:
    """
    Load the first hdu from each path in frame list.
    :param frame_path: A pathlib.Path array of fits files.
    :return: An astropy.io.fits.PrimaryHDU array the same size as frame path
    """

    @np.vectorize
    def load_hdu_scalar(p: pathlib.Path):
        f = astropy.io.fits.open(p)
        f.verify('fix')
        h = f[hdu_index]

        return h

    return load_hdu_scalar(frame_path)


def extract_wcs(hdu: np.ndarray) -> astropy.wcs.WCS:
    def v_func(h: astropy.io.fits.PrimaryHDU): return astropy.wcs.WCS(h.header)

    vec = np.vectorize(v_func)
    a = np.array(vec(hdu))

    return a


def extract_data(hdu: np.ndarray) -> np.ndarray:
    """
    Assemble all the data in an array of hdu objects into a single array
    :param hdu: An astropy.io.fits.PrimaryHDU array with shape (n, ..., m).
    :return: An array composed of all the hdu.data elements with shape (n, ..., m, i, ..., j), where (i, ..., j) is the
    shape of the data arrays.
    """

    def v_func(h: astropy.io.fits.PrimaryHDU): return h.data

    vec = np.vectorize(v_func, signature='()->(m,n)')

    a = np.array(vec(hdu))

    return a.astype(np.float)

def extract_header_value(hdu: np.ndarray, key: str) -> np.ndarray:
    """
    From an array of HDUs and a fits header keyword, make an array with the value of the keyword as each element.
    :param hdu: An `astropy.io.fits.PrimaryHDU` array
    :param key: A fits keyword
    :return: An array of the value of the fits keyword for every element in `hdu`.
    """

    def vec_func(h: astropy.io.fits.PrimaryHDU): return h.header[key]

    vec = np.vectorize(vec_func)

    return np.array(vec(hdu))


def extract_times(hdu: np.ndarray, keyword: str):
    """
    From an array of HDUs, create a time object where each element is a HDU timestamp
    :param hdu: An astropy.io.fits.PrimaryHDU array
    :return: astropy.time.Time object with the timestamps extracted from the HDU array.
    """

    t = extract_header_value(hdu, keyword)

    return astropy.time.Time(t)