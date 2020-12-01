import pytest
import pathlib
import astropy.io.fits


@pytest.mark.skip('Not yet implemented, error in fits file')
def test_cds():
    f0 = pathlib.Path('C:\\Users\\byrdie\\Kankelborg-Group\\kgpy\\kgpy\\observatories\\cds\\data\\s631r00.fits')
    f1 = pathlib.Path('C:\\Users\\byrdie\\Kankelborg-Group\\kgpy\\kgpy\\observatories\\cds\\data\\s631r00.fits')
    f2 = pathlib.Path('C:\\Users\\byrdie\\Kankelborg-Group\\kgpy\\kgpy\\observatories\\cds\\data\\s631r00.fits')

    hdul = astropy.io.fits.open(f1, lazy_load_hdus=True, output_verify='fix')

    print(hdul[1].data.shape)