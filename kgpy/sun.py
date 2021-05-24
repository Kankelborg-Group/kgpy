import typing as typ
import pickle
import pathlib
import astropy.units as u
import pandas
import ChiantiPy.core

__all__ = ['angular_radius_max']

# https://en.wikipedia.org/wiki/Angular_diameter#Use_in_astronomy
angular_radius_max = (32 * u.arcmin + 32 * u.arcsec) / 2  # type: u.Quantity
"""maximum angular radius of the solar disk"""


def dem(dem_file: pathlib.Path) -> typ.Tuple[u.Quantity, u.Quantity]:
    dem = pandas.read_csv(
        filepath_or_buffer=dem_file,
        sep=' ',
        skipinitialspace=True,
        skipfooter=9,
        names=['logT', 'logEM'],
    )
    temperature = 10 ** dem['logT'].to_numpy() * u.K
    em = 10 ** dem['logEM'].to_numpy() * u.dimensionless_unscaled
    return temperature, em


def dem_qs() -> typ.Tuple[u.Quantity, u.Quantity]:
    dem_file = pathlib.Path(r'C:\Users\royts\chianti\dem\quiet_sun.dem')
    return dem(dem_file)


def spectrum_qs_tr() -> ChiantiPy.core.bunch:
    spectrum_qs_tr_cache = pathlib.Path(__file__).parent / 'spectrum_qs_tr_cache.pickle'
    if not spectrum_qs_tr_cache.exists():
        temperature, emission = dem_qs()
        pressure = 1e15 * u.K * u.cm ** -3
        density_electron = pressure / temperature
        bunch = ChiantiPy.core.bunch(
            temperature=temperature.value,
            eDensity=density_electron.value,
            wvlRange=[10, 1000],
            minAbund=1e-5,
            em=emission,
            abundance='sun_coronal_2012_schmelz',
            keepIons=True,
        )
        with open(spectrum_qs_tr_cache, 'wb') as f:
            pickle.dump(bunch, f)

    else:
        with open(spectrum_qs_tr_cache, 'rb') as f:
            bunch = pickle.load(f)

    return bunch
