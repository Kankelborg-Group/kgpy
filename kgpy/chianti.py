import typing as typ
import os
import pathlib
import csv
import numpy as np
import numpy.typing
import matplotlib.axes
import matplotlib.collections
import matplotlib.text
import matplotlib.lines
import roman
import scipy.interpolate
import astropy.units as u
import astropy.visualization
import pandas
import ChiantiPy.core
import adjustText
import kgpy.mixin
import kgpy.format

__all__ = [
    'Bunch',
    'to_spectroscopic',
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update()

    def update(self) -> typ.NoReturn:
        self._indices_masked_sorted = None
        self._ion = None
        self._intensity = None
        self._wavelength = None

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
        intensity = np.trapz(self.Intensity['intensity'], self.temperature[..., np.newaxis], axis=0)
        return intensity * u.erg / u.cm ** 2 / u.s / u.sr

    @property
    def wavelength_mask(self) -> np.ndarray:
        return (self.wavelength_all > self.wavelength_min) & (self.wavelength_all < self.wavelength_max)

    @property
    def indices_masked_sorted(self) -> np.ndarray:
        if self._indices_masked_sorted is None:
            self._indices_masked_sorted = np.argsort(self.intensity_all[self.wavelength_mask])
        return self._indices_masked_sorted

    def _mask_and_sort(self, arr: numpy.typing.ArrayLike) -> numpy.typing.ArrayLike:
        return arr[self.wavelength_mask][self.indices_masked_sorted][::-1]

    @property
    def ion(self) -> np.ndarray:
        if self._ion is None:
            self._ion = self._mask_and_sort(self.ion_all)
        return self._ion

    @property
    def ion_spectroscopic(self):
        return to_spectroscopic(self.ion, use_latex=True)

    @property
    def wavelength(self) -> u.Quantity:
        if self._wavelength is None:
            self._wavelength = self._mask_and_sort(self.wavelength_all)
        return self._wavelength

    @property
    def intensity(self) -> u.Quantity:
        if self._intensity is None:
            self._intensity = self._mask_and_sort(self.intensity_all)
        return self._intensity

    def fullname(self, digits_after_decimal: int = 3, use_latex: bool = True) -> np.ndarray:
        k = dict(digits_after_decimal=digits_after_decimal, scientific_notation=False)
        ion = to_spectroscopic(self.ion, use_latex=use_latex)
        result = [i + ' ' + kgpy.format.quantity(w, **k) for i, w in zip(ion, self.wavelength)]
        return np.array(result)

    def dataframe(self, num_emission_lines: int = 10) -> pandas.DataFrame:
        wavelength = self.wavelength[:num_emission_lines]
        intensity = self.intensity[:num_emission_lines]
        ion = to_spectroscopic(self.ion[:num_emission_lines], use_latex=False)
        return pandas.DataFrame(
            data=[
                ion,
                [kgpy.format.quantity(w) for w in wavelength],
                intensity / intensity.max()
            ],
            index=['ion', 'wavelength', 'intensity']
        ).T

    def plot(
            self,
            ax: matplotlib.axes.Axes,
            num_emission_lines=None,
            num_labels: int = None,
            digits_after_decimal: int = 3,
            relative_int=False,
            force_points=(0.01, 3),
            line_mask=slice(None, None),
            label_fontsize: typ.Union[str, int] = 'small',
            use_latex = True

    ) -> typ.Tuple[matplotlib.collections.LineCollection, typ.List[matplotlib.text.Text]]:
        with astropy.visualization.quantity_support():
            wavelength = self.wavelength[:num_emission_lines]
            intensity = self.intensity[:num_emission_lines]

            if num_labels == None:
                num_labels = num_emission_lines

            if relative_int:
                intensity /= intensity.max()
            ion = to_spectroscopic(self.ion[:num_emission_lines], use_latex=False)
            fullname = self.fullname(digits_after_decimal=digits_after_decimal,use_latex=use_latex)[:num_emission_lines]

            if line_mask != slice(None, None):
                wavelength = wavelength[line_mask]
                intensity = intensity[line_mask]
                fullname = fullname[line_mask]

            lines = ax.vlines(
                x=wavelength,
                ymin=0,
                ymax=intensity,
            )
            ax.set_xlabel(f'wavelength ({ax.get_xlabel()})')
            ax.set_ylabel(f'radiance ({ax.get_ylabel()})')
            text = []
            ha = 'right'
            va = 'top'
            virtual_y = []
            virtual_x = []
            for i in range(num_labels):
                text.append(ax.text(
                    x=wavelength[i],
                    y=intensity[i],
                    # s=ion[i] + ' ' + kgpy.format.quantity(wavelength[i], digits_after_decimal=1),
                    s=fullname[i],
                    # rotation='vertical',
                    ha=ha,
                    va=va,
                    fontsize=label_fontsize,
                ))
                num = int(100 * intensity[i] / intensity[0])
                vy = np.linspace(start=0, stop=intensity[i], num=num, axis=0)
                virtual_y.append(vy)
                virtual_x.append(np.broadcast_to(wavelength[i], vy.shape, subok=True))

            virtual_x = np.concatenate(virtual_x)
            virtual_y = np.concatenate(virtual_y)

            # ax.scatter(virtual_x, virtual_y)
            adjustText.adjust_text(
                texts=text,
                ax=ax,
                arrowprops=dict(
                    arrowstyle='-',
                    connectionstyle='arc3',
                    alpha=0.5,
                    linewidth=0.5,
                ),
                # ha=ha,
                # va=va,
                x=virtual_x.value,
                y=virtual_y.value,
                force_points=force_points,
            )
        return lines, text

    def plot_wavelength(
            self,
            ax: matplotlib.axes.Axes,
            num_emission_lines: int = 10,
            digits_after_decimal: int = 3,
            colors: typ.Optional[typ.Sequence[str]] = None,
    ) -> typ.List[matplotlib.lines.Line2D]:
        if colors is None:
            colors = num_emission_lines * ['black']
        with astropy.visualization.quantity_support():
            wavelength = self.wavelength[:num_emission_lines]
            fullname = self.fullname(digits_after_decimal=digits_after_decimal, use_latex=True)[:num_emission_lines]
            # ax.set_ylabel('{0:latex_inline}'.format(intensity.unit))
            lines = []

            for i in range(wavelength.shape[0]):
                lines.append(ax.axvline(
                    x=wavelength[i],
                    label=fullname[i],
                    linestyle='dashed',
                    color=colors[i],
                ))

        return lines


def to_spectroscopic(ions: typ.Sequence[str], use_latex: bool = True) -> np.ndarray:
    ion_latex = []
    for ion in ions:
        element, ion = ion.split('_')

        element = list(element)
        element[0] = element[0].upper()
        element = ''.join(element)
        ion = roman.toRoman(int(ion))

        if use_latex:
            ion = ion.lower()
            ion = r'\,\textsc{' + ion + '}'
        else:
            ion = ' ' + ion

        ion_latex.append(element + ion)

    return numpy.array(ion_latex)


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


dem_qs_file = 'quiet_sun.dem'

abundance_qs_tr_file = 'sun_coronal_2012_schmelz'

pressure_qs = 1e15 * u.K * u.cm ** -3


def dem_qs() -> u.Quantity:
    dem_file = pathlib.Path(os.environ['XUVTOP']) / 'dem' / dem_qs_file
    return dem(dem_file)


def bunch_tr(emission_measure: u.Quantity) -> Bunch:
    bunch_tr_cache = pathlib.Path(__file__).parent / 'bunch_tr_cache.pickle'

    if not bunch_tr_cache.exists():
        temp = temperature()
        density_electron = pressure_qs / temp
        bunch = kgpy.chianti.Bunch(
            temperature=temp.value,
            eDensity=density_electron.value,
            wvlRange=[10, 1000],
            minAbund=1e-5,
            abundance=abundance_qs_tr_file,
        )
        bunch.to_pickle(bunch_tr_cache)

    else:
        bunch = kgpy.chianti.Bunch.from_pickle(bunch_tr_cache)

    bunch.Intensity['intensity'] = bunch.Intensity['intensity'] * emission_measure[..., np.newaxis]

    return bunch


def bunch_tr_qs() -> Bunch:
    return bunch_tr(emission_measure=dem_qs())
