import dataclasses
import typing as typ
import numpy as np
import numpy.typing
import matplotlib.axes
import matplotlib.collections
import matplotlib.text
import astropy.units as u
import astropy.visualization
import pandas
import kgpy.mixin
import kgpy.format

__all__ = ['Transition']


@dataclasses.dataclass
class Transition(kgpy.mixin.Dataframable):
    ion: numpy.typing.ArrayLike
    wavelength: u.Quantity
    intensity: u.Quantity

    @property
    def dataframe(self) -> pandas.DataFrame:
        return pandas.DataFrame(
            data=[
                self.ion,
                [kgpy.format.quantity(w) for w in self.wavelength],
                self.intensity / self.intensity.max()
            ],
            index=['ion', 'wavelength', 'intensity']
        ).T

    def plot(
            self,
            ax: matplotlib.axes.Axes,
    ) -> typ.Tuple[matplotlib.collections.LineCollection, typ.List[matplotlib.text.Text]]:
        with astropy.visualization.quantity_support():
            lines = ax.vlines(
                x=self.wavelength,
                ymin=0,
                ymax=self.intensity,
            )
            text = []
            for i in range(self.wavelength.size):
                index = np.unravel_index(i, shape=self.wavelength.size)
                text.append(ax.text(
                    x=self.wavelength[index],
                    y=self.intensity[index],
                    s=' ' + self.ion[index] + ' ' + str(self.wavelength[index].value),
                    rotation=90,
                    ha='center',
                    va='bottom',
                ))
                # ax.text(wavelength_qs[i], intensity_qs[i], ' ' + ion_qs[i] + ' ' + str(wavelength_qs[i].value),
                #         ha='center', va='bottom', rotation=90)
        return lines, text
