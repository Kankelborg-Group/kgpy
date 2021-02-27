"""
Interfaces for various solar observatories.
"""
import typing as typ
import pathlib
import dataclasses
import numpy as np
import scipy.stats
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.animation
import matplotlib.colors
import matplotlib.lines
import matplotlib.dates
import matplotlib.ticker
import astropy.units as u
import astropy.time
import astropy.wcs
import astropy.visualization
from kgpy import mixin, plot

__all__ = ['Image', 'ImageAxis', 'spectral']


class ImageAxis(mixin.AutoAxis):
    def __init__(self):
        super().__init__()
        self.x = self.auto_axis_index()
        self.y = self.auto_axis_index()
        self.time = self.auto_axis_index(from_right=False)
        self.channel = self.auto_axis_index(from_right=False)

    @property
    def xy(self) -> typ.Tuple[int, int]:
        return self.x, self.y


@dataclasses.dataclass
class Image(mixin.Pickleable):
    axis: typ.ClassVar[ImageAxis] = ImageAxis()                 #: Relationship between physical dimension and axis index.
    intensity: typ.Optional[u.Quantity] = None              #: Intensity of each pixel in the data
    intensity_uncertainty: typ.Optional[u.Quantity] = None
    wcs: typ.Optional[np.ndarray] = None
    time: typ.Optional[astropy.time.Time] = None
    time_index: typ.Optional[np.ndarray] = None
    channel: typ.Optional[u.Quantity] = None
    exposure_length: typ.Optional[u.Quantity] = None

    def __post_init__(self):
        self._time_to_index_cache = None
        self._index_to_time_cache = None

    @classmethod
    def zeros(cls, shape: typ.Sequence[int]) -> 'Image':
        sh = shape[:cls.axis.num_left_dim]
        self = cls()
        self.intensity = np.zeros(shape) * u.adu
        self.intensity_uncertainty = np.zeros(shape) * u.adu
        self.wcs = np.empty(sh, dtype=astropy.wcs.WCS)
        self.time = astropy.time.Time(np.zeros(sh), format='unix')
        self.time_index = np.arange(shape[self.axis.time])
        self.channel = np.zeros(shape[1], dtype=np.int) * u.chan
        self.exposure_length = np.zeros(sh) * u.s
        return self

    @staticmethod
    def default_pickle_path() -> pathlib.Path:
        return pathlib.Path('cube.pickle')

    @property
    def time_exp_start(self) -> astropy.time.Time:
        return self.time - self.exposure_length / 2

    @property
    def time_exp_end(self) -> astropy.time.Time:
        return self.time + self.exposure_length / 2

    @property
    def shape(self) -> typ.Tuple[int, ...]:
        return self.intensity.shape

    @property
    def num_times(self) -> int:
        return self.shape[self.axis.time]

    @property
    def num_channels(self) -> int:
        return self.shape[self.axis.channel]

    @property
    def num_x(self) -> int:
        return self.shape[self.axis.x]

    @property
    def num_y(self) -> int:
        return self.shape[self.axis.y]

    @property
    def channel_labels(self) -> typ.List[str]:
        # return ['Ch' + str(int(c.value)) for c in self.channel[0]]
        return ['ch' + str(int(c.value)) for c in self.channel]

    # @property
    # def time_mid(self) -> astropy.time.Time:
    #     return self.time + self.exposure_length / 2

    # @property
    # def _time_to_index_params(self):
    #     if self._time_to_index_parameters is None:
    #         self._time_to_index_parameters = scipy.stats.linregress(self.time[:, 0].to_value('mjd'), self.time_index, )
    #     return self._time_to_index_parameters
    #
    # @property
    # def _index_to_time_params(self):
    #     if self._index_to_time_parameters is None:
    #         self._index_to_time_parameters = scipy.stats.linregress(self.time_index, self.time[:, 0].to_value('mjd'), )
    #     return self._index_to_time_parameters

    @property
    def _time_to_index(self) -> typ.Callable[[np.ndarray], np.ndarray]:
        t0 = matplotlib.dates.date2num(self.time_exp_start.min(axis=~0).to_datetime())
        t1 = matplotlib.dates.date2num(self.time_exp_end.min(axis=~0).to_datetime())
        time = np.stack([t0, t1]).flatten()
        index = np.stack([self.time_index, self.time_index + 1]).flatten()
        if self._time_to_index_cache is None:
            self._time_to_index_cache = scipy.interpolate.interp1d(
                x=time,
                y=index,
                fill_value='extrapolate',
            )
        return self._time_to_index_cache

    @property
    def _index_to_time(self):
        t0 = matplotlib.dates.date2num(self.time_exp_start.min(axis=~0).to_datetime())
        t1 = matplotlib.dates.date2num(self.time_exp_end.min(axis=~0).to_datetime())
        time = np.stack([t0, t1]).flatten()
        index = np.stack([self.time_index, self.time_index + 1]).flatten()
        if self._index_to_time_cache is None:
            self._index_to_time_cache = scipy.interpolate.interp1d(
                x=index,
                y=time,
                fill_value='extrapolate',
            )
        return self._index_to_time_cache

    def plot_quantity_vs_index(
            self,
            a: u.Quantity,
            a_name: str = '',
            ax: typ.Optional[plt.Axes] = None,
            legend_ncol: int = 1,
            drawstyle: str = 'steps-mid',
    ) -> plt.Axes:
        """

        Parameters
        ----------
        a:
        a_name
        ax :
        legend_ncol
        drawstyle

        Returns
        -------
        matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots()

        # locator = matplotlib.dates.AutoDateLocator()
        # locator.intervald[dateutil.rrule.MINUTELY] = [1, 2, 5, 10, 15, 20, 30]
        # formatter = matplotlib.dates.AutoDateFormatter(locator=locator)
        # formatter.scaled[(1 * u.min).to(u.day).value] = '%H:%M:%S'
        # ax.xaxis.set_major_formatter(formatter)
        # ax.xaxis.set_major_locator(locator)
        # ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax = plot.datetime_prep(ax)

        ax2 = ax.secondary_xaxis(
            location='top',
            functions=(self._time_to_index, self._index_to_time),
        )
        ax2.set_xlabel('exposure index')

        with astropy.visualization.quantity_support():
            # with astropy.visualization.time_support(format='isot'):
            # ax.xaxis.set_tick_params(rotation=90)

            for c in range(self.num_channels):
                ax.plot(
                    self.time[:, c].to_datetime(),
                    a[:, c],
                    label=a_name + ', ' + self.channel_labels[c],
                    drawstyle=drawstyle,
                )


            # ax.figure.autofmt_xdate()
            # ax.legend(fontsize='small', ncol=legend_ncol, loc='center left', bbox_to_anchor=(1, 0.5))
            # ax.legend()
        return ax

    def plot_intensity_mean_vs_time(self, ax: typ.Optional[plt.Axes] = None, ) -> plt.Axes:
        return self.plot_quantity_vs_index(
            a=self.intensity.mean(self.axis.xy), a_name='Mean intensity', ax=ax)

    def plot_exposure_length(self, ax: typ.Optional[plt.Axes] = None, ) -> plt.Axes:
        return self.plot_quantity_vs_index(a=self.exposure_length, a_name='exposure length', ax=ax)

    def plot_channel(
            self,
            image: u.Quantity,
            image_name: str = '',
            ax: typ.Optional[plt.Axes] = None,
            thresh_min: u.Quantity = 0.01 * u.percent,
            thresh_max: u.Quantity = 99.9 * u.percent,
            colorbar_location: str = 'right',
            transpose: bool = False
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.set_title(image_name)

        label_x = 'detector $x$ (pix)'
        label_y = 'detector $y$ (pix)'

        if transpose:
            image = image.T
            label_x, label_y = label_y, label_x

        img = ax.imshow(
            X=image.value,
            vmin=np.percentile(image, thresh_min.value).value,
            vmax=np.percentile(image, thresh_max.value).value,
            origin='lower',
        )
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        fig.colorbar(img, ax=ax, label=image.unit, location=colorbar_location, use_gridspec=False)
        return ax

    def plot_channel_from_data(
            self,
            data: u.Quantity,
            ax: typ.Optional[plt.Axes] = None,
            time_index: int = 0,
            channel_index: int = 0,
            thresh_min: u.Quantity = 0.01 * u.percent,
            thresh_max: u.Quantity = 99.9 * u.percent,
    ) -> plt.Axes:
        image = data[time_index, channel_index]
        time = self.time[time_index, channel_index]
        chan = self.channel[channel_index]
        seq_index = self.time_index[time_index]
        image_name = time.to_value('iso') + ', frame ' + str(int(seq_index)) + ', channel ' + str(int(chan.value))
        return self.plot_channel(
            image=image, image_name=image_name, ax=ax, thresh_max=thresh_max, thresh_min=thresh_min
        )

    def plot_intensity_channel(
            self,
            ax: typ.Optional[plt.Axes] = None,
            time_index: int = 0,
            channel_index: int = 0,
            thresh_min: u.Quantity = 0.01 * u.percent,
            thresh_max: u.Quantity = 99.9 * u.percent,
    ) -> plt.Axes:
        return self.plot_channel_from_data(
            data=self.intensity, ax=ax,
            time_index=time_index, channel_index=channel_index,
            thresh_max=thresh_max, thresh_min=thresh_min
        )

    def plot_time(
            self,
            images: u.Quantity,
            image_names: typ.Sequence[str],
            axs: typ.Optional[typ.MutableSequence[plt.Axes]],
            thresh_min: u.Quantity = 0.01 * u.percent,
            thresh_max: u.Quantity = 99.9 * u.percent,
    ) -> np.ndarray:

        if axs is None:
            fig, axs = plt.subplots(ncols=self.num_channels, squeeze=False)

        i = 0
        for ax, image, image_name in zip(axs.flatten(), images, image_names):
            self.plot_channel(
                image=image, image_name=image_name, ax=ax, thresh_max=thresh_max, thresh_min=thresh_min,
                colorbar_location='bottom', transpose=True,
            )
            if i != 0:
                ax.set_ylabel(None)
            i += 1
        return axs

    def plot_time_from_data(
            self,
            data: u.Quantity,
            axs: typ.Optional[typ.MutableSequence[plt.Axes]] = None,
            time_index: int = 0,
            thresh_min: u.Quantity = 0.01 * u.percent,
            thresh_max: u.Quantity = 99.9 * u.percent,
    ) -> np.ndarray:
        time = self.time[time_index, 0]
        seq_index = self.time_index[time_index]
        axs = self.plot_time(
            images=data[time_index],
            image_names=self.channel_labels,
            axs=axs,
            thresh_max=thresh_max,
            thresh_min=thresh_min,
        )
        axs[0, 0].figure.suptitle(time.to_value('iso') + ', frame ' + str(int(seq_index)))
        return axs

    def plot_intensity_time(
            self,
            axs: typ.Optional[typ.MutableSequence[plt.Axes]] = None,
            time_index: int = 0,
            thresh_min: u.Quantity = 0.01 * u.percent,
            thresh_max: u.Quantity = 99.9 * u.percent,
    ) -> np.ndarray:
        return self.plot_time_from_data(
            data=self.intensity, axs=axs, time_index=time_index, thresh_max=thresh_max, thresh_min=thresh_min
        )

    def animate_channel(
            self,
            images: u.Quantity,
            image_names: typ.List[str],
            ax: typ.Optional[plt.Axes] = None,
            thresh_min: u.Quantity = 0.01 * u.percent,
            thresh_max: u.Quantity = 99.9 * u.percent,
            norm_gamma: float = 1,
            norm_vmin: typ.Optional[u.Quantity] = None,
            norm_vmax: typ.Optional[u.Quantity] = None,
            frame_interval: u.Quantity = 1 * u.s,
    ):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        vmin, vmax = norm_vmin, norm_vmax
        if vmin is None:
            vmin = np.percentile(images[0], thresh_min.value)
        if norm_vmax is None:
            vmax = np.percentile(images[0], thresh_max.value)
        img = ax.imshow(
            X=images[0].value,
            norm=matplotlib.colors.PowerNorm(gamma=norm_gamma),
            vmin=vmin.value,
            vmax=vmax.value,
            origin='lower',
        )

        title = ax.set_title(image_names[0])
        ax.set_xlabel('detector $x$ (pix)')
        ax.set_ylabel('detector $y$ (pix)')
        fig.colorbar(img, ax=ax, label=images.unit, fraction=0.05)

        def func(i: int):
            img.set_data(images[i].value)
            title.set_text(image_names[i])
            if norm_vmin is None:
                img.set_clim(vmin=np.percentile(images[i], thresh_min.value).value)
            if norm_vmax is None:
                img.set_clim(vmax=np.percentile(images[i], thresh_max.value).value)

        fig.set_constrained_layout(False)

        return matplotlib.animation.FuncAnimation(
            fig=fig,
            func=func,
            frames=images.shape[0],
            interval=frame_interval.to(u.ms).value,
        )

    def animate_intensity_channel(
            self,
            ax: typ.Optional[plt.Axes] = None,
            time_slice: typ.Optional[slice] = None,
            channel_index: int = 0,
            thresh_min: u.Quantity = 0.01 * u.percent,
            thresh_max: u.Quantity = 99.9 * u.percent,
            norm_gamma: float = 1,
            frame_interval: u.Quantity = 100 * u.ms,
    ) -> matplotlib.animation.FuncAnimation:

        if time_slice is None:
            time_slice = slice(None)

        def title_text(i: int) -> str:
            time_str = self.time[i, channel_index].to_value('iso')
            channel_str = ', channel ' + str(int(self.channel[channel_index].value))
            frame_str = ', frame ' + str(int(self.time_index[i]))
            return time_str + frame_str + channel_str

        images = self.intensity[time_slice, channel_index]
        return self.animate_channel(
            images=images,
            image_names=[title_text(i) for i in range(self.num_times)][time_slice],
            ax=ax,
            thresh_min=thresh_min,
            thresh_max=thresh_max,
            norm_gamma=norm_gamma,
            norm_vmin=np.percentile(images, thresh_min.value),
            norm_vmax=np.percentile(images, thresh_max.value),
            frame_interval=frame_interval,
        )

    def animate(
            self,
            data: u.Quantity,
            time_slice: slice = slice(None),
            axs: typ.Optional[np.ndarray] = None,
            thresh_min: u.Quantity = 0.01 * u.percent,
            thresh_max: u.Quantity = 99.9 * u.percent,
            norm_gamma: float = 1,
            frame_interval: u.Quantity = 100 * u.ms,
    ) -> matplotlib.animation.FuncAnimation:

        if axs is None:
            fig, axs = plt.subplots(ncols=self.num_channels, squeeze=False)
        else:
            fig = axs[0, 0].figure

        data = data[time_slice]
        start_time = self.time[time_slice]
        sequence_index = self.time_index[time_slice]

        def title_text(i: int) -> str:
            time_str = start_time[i, 0].to_value('iso')
            frame_str = ', frame ' + str(int(sequence_index[i]))
            return time_str + frame_str

        title = fig.suptitle(title_text(0))

        imgs = []
        for c in range(self.num_channels):

            ax = axs.flatten()[c]
            ax.set_title(self.channel_labels[c])

            label_x = 'detector $x$ (pix)'
            label_y = 'detector $y$ (pix)'
            label_x, label_y = label_y, label_x

            image = data[0, c].T

            img = ax.imshow(
                X=image.value,
                vmin=np.percentile(data[:, c], thresh_min.value).value,
                vmax=np.percentile(data[:, c], thresh_max.value).value,
                origin='lower',
                norm=matplotlib.colors.PowerNorm(norm_gamma)
            )
            imgs.append(img)
            ax.set_xlabel(label_x)
            if c == 0:
                ax.set_ylabel(label_y)
            fig.colorbar(img, ax=axs[0, c], label=image.unit, location='bottom', use_gridspec=False)

        def func(i: int):
            for c in range(self.num_channels):
                imgs[c].set_data(data[i, c].T.value)
                title.set_text(title_text(i))

        # fig.set_constrained_layout(False)

        return matplotlib.animation.FuncAnimation(
            fig=fig,
            func=func,
            frames=data.shape[0],
            interval=frame_interval.to(u.ms).value,
        )

    def animate_intensity(
            self,
            axs: typ.Optional[typ.MutableSequence[plt.Axes]] = None,
            thresh_min: u.Quantity = 0.01 * u.percent,
            thresh_max: u.Quantity = 99.9 * u.percent,
            norm_gamma: float = 1,
            frame_interval: u.Quantity = 100 * u.ms,
    ) -> matplotlib.animation.FuncAnimation:
        return self.animate(
            data=self.intensity,
            axs=axs, thresh_min=thresh_min, thresh_max=thresh_max, norm_gamma=norm_gamma, frame_interval=frame_interval
        )

from . import spectral
