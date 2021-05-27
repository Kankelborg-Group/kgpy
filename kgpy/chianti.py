import typing as typ
import os
import pathlib
import csv
import numpy as np
import scipy.interpolate
import astropy.units as u
import ChiantiPy.core as ch
import kgpy.mixin
from collections import OrderedDict



class Bunch(kgpy.mixin.Pickleable, ch.bunch):
    pass

def write_roman(num):

    roman = OrderedDict()
    roman[40] = "xl"
    roman[10] = "x"
    roman[9] = "ix"
    roman[5] = "v"
    roman[4] = "iv"
    roman[1] = "i"

    def roman_num(num):
        for r in roman.keys():
            x, y = divmod(num, r)
            yield roman[r] * x
            num -= (r * x)
            if num <= 0:
                break

    return "".join([a for a in roman_num(num)])

def ion_tolatex(ions):
    ion_latex = []
    for ion in ions:
        element, ion = ion.split('_')
        # ion_latex.append(element[0].upper()+element[1:]+'\,{\sc '+write_roman(int(ion))+'}')
        ion_latex.append(element[0].upper()+element[1:]+'\,$\textsc{'+write_roman(int(ion))+'}$')
    return ion_latex


def temperature() -> u.Quantity:
    return 10 ** np.arange(4, 8.1, 0.1) * u.K


def dem(file: pathlib.Path) -> u.Quantity:
    temp = []
    emission = []
    with open(file, newline='') as fp:
        dem_reader = csv.reader(fp, delimiter=' ', skipinitialspace=True)
        for row in dem_reader:
            if float(row[0]) == -1:
                break
            temp.append(10 ** float(row[0]))
            emission.append(10 ** float(row[1]))

    temp = temp * u.K
    emission = emission * u.dimensionless_unscaled

    emission_interp = scipy.interpolate.interp1d(
        x=temp,
        y=emission,
        fill_value=0,
        bounds_error=False,
    )

    return emission_interp(temperature())


def dem_qs() -> u.Quantity:
    dem_file = pathlib.Path(os.environ['XUVTOP']) / 'dem/quiet_sun.dem'
    return dem(dem_file)


def bunch_tr(emission_measure: u.Quantity) -> Bunch:
    bunch_tr_cache = pathlib.Path(__file__).parent / 'bunch_tr_cache.pickle'

    if not bunch_tr_cache.exists():
        temp = temperature()
        pressure = 1e15 * u.K * u.cm ** -3
        density_electron = pressure / temp
        bunch = kgpy.chianti.Bunch(
            temperature=temp.value,
            eDensity=density_electron.value,
            wvlRange=[10, 1000],
            minAbund=1e-5,
            abundance='sun_coronal_2012_schmelz',
        )
        bunch.to_pickle(bunch_tr_cache)

    else:
        bunch = kgpy.chianti.Bunch.from_pickle(bunch_tr_cache)

    bunch.Intensity['intensity'] = bunch.Intensity['intensity'] * emission_measure[..., np.newaxis]

    return bunch


def bunch_tr_qs() -> Bunch:
    return bunch_tr(emission_measure=dem_qs())
