import typing as typ
import os
import pathlib
import csv
import numpy as np
import numpy.typing
import matplotlib.axes
import matplotlib.collections
import matplotlib.text
import scipy.interpolate
import astropy.units as u
import astropy.visualization
import pandas
import ChiantiPy.core
import kgpy.mixin
import kgpy.format
from collections import OrderedDict

__all__ = [
    'Bunch',
    'write_roman',
    'ion_tolatex',
    'temperature',
    'dem',
    'dem_qs',
    'bunch_tr',
    'bunch_tr_qs',
]


class Bunch(
    kgpy.mixin.Pickleable,
    ChiantiPy.core.bunch,
):

    @property
    def wavelength_min(self) -> u.Quantity:
        return self.WvlRange[0] << u.AA

    @wavelength_min.setter
    def wavelength_min(self, value: u.Quantity):
        self.WvlRange[0] = value.to(u.AA).value

    @property
    def wavelength_max(self) -> u.Quantity:
        return self.WvlRange[1] << u.AA

    @wavelength_max.setter
    def wavelength_max(self, value: u.Quantity):
        self.WvlRange[1] = value.to(u.AA).value

    @property
    def temperature(self) -> u.Quantity:
        return self.Temperature

    @property
    def ion_all(self) -> np.ndarray:
        return self.Intensity['ionS']

    @property
    def wavelength_all(self) -> u.Quantity:
        return self.Intensity['wvl'] << u.AA

    @property
    def intensity_all(self) -> u.Quantity:
        return np.trapz(self.Intensity['intensity'], self.temperature[..., np.newaxis], axis=0) << u.dimensionless_unscaled

    @property
    def wavelength_mask(self) -> np.ndarray:
        return (self.wavelength_all > self.wavelength_min) & (self.wavelength_all < self.wavelength_max)

    @property
    def indices_masked_sorted(self) -> np.ndarray:
        return np.argsort(self.intensity_all[self.wavelength_mask])

    def _mask_and_sort(self, arr: numpy.typing.ArrayLike) -> numpy.typing.ArrayLike:
        return arr[self.wavelength_mask][self.indices_masked_sorted][::-1]

    @property
    def ion(self) -> np.ndarray:
        return self._mask_and_sort(self.ion_all)

    @property
    def wavelength(self) -> u.Quantity:
        return self._mask_and_sort(self.wavelength_all)

    @property
    def intensity(self) -> u.Quantity:
        return self._mask_and_sort(self.intensity_all)

    def dataframe(self, num_emission_lines: int = 10) -> pandas.DataFrame:
        wavelength = self.wavelength[:num_emission_lines]
        intensity = self.intensity[:num_emission_lines]
        return pandas.DataFrame(
            data=[
                self.ion[:num_emission_lines],
                [kgpy.format.quantity(w) for w in wavelength],
                intensity / intensity.max()
            ],
            index=['ion', 'wavelength', 'intensity']
        ).T

    def plot(
            self,
            ax: matplotlib.axes.Axes,
            num_emission_lines: int = 10
    ) -> typ.Tuple[matplotlib.collections.LineCollection, typ.List[matplotlib.text.Text]]:

        with astropy.visualization.quantity_support():
            wavelength = self.wavelength[:num_emission_lines]
            intensity = self.intensity[:num_emission_lines]
            ion = self.ion[:num_emission_lines]
            lines = ax.vlines(
                x=wavelength,
                ymin=0,
                ymax=intensity,
            )
            text = []
            for i in range(wavelength.shape[0]):
                text.append(ax.text(
                    x=wavelength[i],
                    y=intensity[i],
                    s=' ' + ion[i] + ' ' + str(wavelength[i].value),
                    rotation=90,
                    ha='center',
                    va='bottom',
                ))
        return lines, text


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
