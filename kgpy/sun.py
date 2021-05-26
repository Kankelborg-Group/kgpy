import typing as typ
import os
import pathlib
import astropy.units as u
import pandas
import kgpy.chianti

__all__ = ['angular_radius_max']

# https://en.wikipedia.org/wiki/Angular_diameter#Use_in_astronomy
angular_radius_max = (32 * u.arcmin + 32 * u.arcsec) / 2  # type: u.Quantity
"""maximum angular radius of the solar disk"""


def dem(dem_file: pathlib.Path) -> typ.Tuple[u.Quantity, u.Quantity]:
    dem_csv = pandas.read_csv(
        filepath_or_buffer=dem_file,
        sep=' ',
        skipinitialspace=True,
        skipfooter=9,
        names=['logT', 'logEM'],
    )
    temperature = 10 ** dem_csv['logT'].to_numpy() * u.K
    em = 10 ** dem_csv['logEM'].to_numpy() * u.dimensionless_unscaled
    return temperature, em


def dem_qs() -> typ.Tuple[u.Quantity, u.Quantity]:
    dem_file = pathlib.Path(os.environ['XUVTOP']) / 'dem/quiet_sun.dem'
    return dem(dem_file)


def spectrum_qs_tr() -> kgpy.chianti.Bunch:
    spectrum_qs_tr_cache = pathlib.Path(__file__).parent / 'spectrum_qs_tr_cache.pickle'
    if not spectrum_qs_tr_cache.exists():
        temperature, emission = dem_qs()
        pressure = 1e15 * u.K * u.cm ** -3
        density_electron = pressure / temperature
        bunch = kgpy.chianti.Bunch(
            temperature=temperature.value,
            eDensity=density_electron.value,
            wvlRange=[10, 1000],
            minAbund=1e-5,
            em=emission,
            abundance='sun_coronal_2012_schmelz',
            keepIons=True,
        )
        bunch.to_pickle(spectrum_qs_tr_cache)

    else:
        bunch = kgpy.chianti.Bunch.from_pickle(spectrum_qs_tr_cache)

    return bunch
