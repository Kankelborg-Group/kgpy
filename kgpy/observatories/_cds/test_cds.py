import pytest
import pathlib
import astropy.io.fits
import matplotlib.pyplot as plt


@pytest.mark.skip('Not yet implemented, error in fits file')
def test_cds():
    f0 = pathlib.Path('C:\\Users\\byrdie\\Kankelborg-Group\\kgpy\\kgpy\\observatories\\cds\\data\\s631r00.fits')
    f1 = pathlib.Path('C:\\Users\\byrdie\\Kankelborg-Group\\kgpy\\kgpy\\observatories\\cds\\data\\s631r00.fits')
    f3 = pathlib.Path('C:\\Users\\byrdie\\Kankelborg-Group\\kgpy\\kgpy\\observatories\\cds\\data\\s9089r19.fits')
    f4 = pathlib.Path(r'C:\Users\byrdie\Kankelborg-Group\kgpy\kgpy\observatories\cds\fdm\s7130r00.fits')

    hdul = astropy.io.fits.open(f4)

    print(repr(hdul[1].header))

    print(type(hdul[1]))

    hdul[1].header['TTYPE7'] = 'BACKGROUND7'
    hdul[1].header['TTYPE8'] = 'BACKGROUND8'
    hdul[1].header['TTYPE9'] = 'BACKGROUND9'
    hdul[1].header['TTYPE10'] = 'BACKGROUND10'

    # print(hdul[1].data[0].columns)

    plt.imshow(hdul[1].data[0]['O_5_629_73'][20])
    plt.show()
