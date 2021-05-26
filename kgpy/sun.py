import pathlib
import astropy.units as u
import kgpy.chianti

__all__ = ['angular_radius_max']

# https://en.wikipedia.org/wiki/Angular_diameter#Use_in_astronomy
angular_radius_max = (32 * u.arcmin + 32 * u.arcsec) / 2  # type: u.Quantity
"""maximum angular radius of the solar disk"""


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
