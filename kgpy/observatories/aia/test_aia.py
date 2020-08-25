import pathlib
from kgpy.observatories.aia import aia
import datetime
import astropy.units as u

def test_fetch_from_time(capsys):

    start_time = datetime.datetime(2019,9,30,0,0,0,0)
    end_time = start_time + datetime.timedelta(seconds = 15)

    path = pathlib.Path(__file__).parent / 'test_jsoc'

    with capsys.disabled():

        print(start_time)
        print(end_time)
        channel = [304*u.angstrom]
        paths = aia.fetch_from_time(start_time, end_time, path, aia_channels= channel)

        return(paths)

def test_prep_from_paths(capsys):
    downloaded_files = test_fetch_from_time(capsys)


    aia.aiaprep_from_paths(downloaded_files)


def test_from_path():
    data_path = pathlib.Path(__file__).parent / 'data'
    path_304 = data_path / '304/lev15'

    aia_304 = aia.AIA.from_path('aia_304',path_304)

    assert aia_304.intensity.shape[0] > 0

