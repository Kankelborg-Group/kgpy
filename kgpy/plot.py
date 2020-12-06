import typing as typ
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent, MouseButton
import astropy.wcs
from kgpy import format as fmt

__all__ = ['ImageSlicer', 'CubeSlicer', 'HypercubeSlicer']


class ImageSlicer:

    def __init__(self, x, y, **kwargs):

        fig, ax = plt.subplots(1, 1)
        self.fig = fig
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        x, y = np.broadcast_arrays(x, y, subok=True)
        self.x = x
        self.y = y
        self.sh = x.shape
        self.ind = 0

        self.plot = ax.plot(self.x[self.ind], self.y[self.ind], **kwargs)
        self.update()

        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = np.clip(self.ind + 1, 0, self.sh[0] - 1)
        else:
            self.ind = np.clip(self.ind - 1, 0, self.sh[0] - 1)
        self.update()

    def update(self):

        self.ax.set_ylabel('slice %s' % self.ind)

        self.plot[0].set_xdata(self.x[self.ind])
        self.plot[0].set_ydata(self.y[self.ind])

        self.fig.canvas.draw()


class CubeSlicer:
    def __init__(self, cube, figsize: typ.Tuple[float, float] = (6, 4), **kwargs):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self.fig = fig
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.cube = cube
        self.sh = cube.shape
        self.ind = 0

        self.im = ax.imshow(self.cube[self.ind], **kwargs)
        self.update()

        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = np.clip(self.ind + 1, 0, self.sh[0] - 1)
        else:
            self.ind = np.clip(self.ind - 1, 0, self.sh[0] - 1)
        self.update()

    def update(self):
        self.im.set_data(self.cube[self.ind,])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

    def set_data(self, X):
        self.cube = X
        self.ind = 0
        self.update()

    def save(self,path: str):
        """
        Method defined to save each image in the cube.  Images are saved in directory "path" in order as i.png
        :param path:
        :return:
        """
        j = self.ind
        if not os.path.exists(path):
            os.makedirs(path)

        for i in range(self.slices):
            self.ind = i
            self.update()
            self.fig.savefig(os.path.join(os.path.dirname(path),str(i)+'.png'))
        self.ind = j


class HypercubeAxis:
    t = ~3
    y = ~2
    x = ~1
    z = ~0


class HypercubeSlicer:
    _i_xy = (0, 0)
    _i_xz = (1, 0)
    _i_yz = (0, 1)
    _i_z = (1, 1)

    _axis = HypercubeAxis()

    line_kwargs = {
        'color': 'red',
        'alpha': 0.5,
    }

    def __init__(
            self,
            data: np.ndarray,
            wcs_list: typ.Optional[typ.List[astropy.wcs.WCS]] = None,
            percentile_thresh: typ.Tuple[float, float] = (.1, 99.9),
            width_ratios: typ.Tuple[float, float] = (1, 1),
            height_ratios: typ.Tuple[float, float] = (1, 1),
    ):

        self._data = data
        self._data_xy_median = np.median(data, axis=(self._axis.x, self._axis.y))
        self._wcs_list = wcs_list
        self.percentile_thresh = percentile_thresh

        sh = data.shape

        fig = plt.figure(constrained_layout=True, figsize=(10, 8))

        gridspec = fig.add_gridspec(
            nrows=2,
            ncols=2,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )

        self._fig = fig  # type: plt.Figure
        self._gridspec = gridspec

        self._x_index = sh[self._axis.x] // 2
        self._y_index = sh[self._axis.y] // 2
        self._z_index = sh[self._axis.z] // 2
        self._t_index = 0
        self._integrate = False

        self._ax_xy = self._fig.add_subplot(
            self._gridspec[self._i_xy],
            projection=self._wcs_xy
        )
        self._ax_xz = self._fig.add_subplot(
            self._gridspec[self._i_xz],
            projection=self._wcs_xz,
            sharex=self._ax_xy,
        )
        self._ax_yz = self._fig.add_subplot(
            self._gridspec[self._i_yz],
            projection=self._wcs_yz,
            sharey=self._ax_xy,
        )
        self._ax_z = self._fig.add_subplot(
            self._gridspec[self._i_z],
            projection=self._wcs_z,
            sharex=self._ax_yz,

        )

        self._vmin = np.percentile(self._data, self.percentile_thresh[0])
        self._vmax = np.percentile(self._data, self.percentile_thresh[~0])

        self._xy_vline = self._ax_xy.axvline(self._x_index_int, **self.line_kwargs)
        self._xy_hline = self._ax_xy.axhline(self._y_index_int, **self.line_kwargs)

        self._xz_vline = self._ax_xz.axvline(self._x_index_int, **self.line_kwargs)
        self._xz_hline = self._ax_xz.axhline(self._z_index_int, **self.line_kwargs)

        self._yz_vline = self._ax_yz.axvline(self._z_index_int, **self.line_kwargs)
        self._yz_hline = self._ax_yz.axhline(self._y_index_int, **self.line_kwargs)

        self._z_vline = self._ax_z.axvline(self._z_index, **self.line_kwargs)

        self._plot_z = self._ax_z.plot(self._data_z)
        self._plot_z_med = self._ax_z.plot(self._data_xy_median[self.t_index])

        self._img_xy = self._ax_xy.imshow(
            X=self._data_xy,
            aspect='auto',
            vmin=self._vmin,
            vmax=self._vmax,
            interpolation='nearest',
            origin='lower'
        )

        self._img_xz = self._ax_xz.imshow(
            self._data_xz,
            aspect='auto',
            vmax=self._vmax,
            vmin=self._vmin,
            interpolation='nearest',
            origin='lower'
        )

        self._img_yz = self._ax_yz.imshow(
            self._data_yz,
            aspect='auto',
            vmax=self._vmax,
            vmin=self._vmin,
            interpolation='nearest',
            origin='lower'
        )

        self.t_index = 0
        self._cid = None

        fig.set_constrained_layout(False)

        fig.canvas.mpl_connect('button_press_event', self._left_click)
        fig.canvas.mpl_connect('button_release_event', self._left_unclick)
        fig.canvas.mpl_connect('scroll_event', self._update_scroll)
        fig.canvas.mpl_connect('key_press_event', self._update_keypress)

    def _left_click(self, event: MouseEvent):
        if event.button is MouseButton.LEFT:
            self._update_mouse(event)
            self._cid = self._fig.canvas.mpl_connect('motion_notify_event', self._update_mouse)

    def _left_unclick(self, event: MouseEvent):
        if event.button is MouseButton.LEFT:
            if self._cid is not None:
                self._fig.canvas.mpl_disconnect(self._cid)

    def _update_mouse(self, event: MouseEvent):
        if event.inaxes is self._ax_xy:
            self.x_index = event.xdata
            self.y_index = event.ydata
        elif event.inaxes is self._ax_xz:
            self.x_index = event.xdata
            self.z_index = event.ydata
        elif event.inaxes is self._ax_yz:
            self.z_index = event.xdata
            self.y_index = event.ydata
        self._fig.canvas.draw()

    def _update_scroll(self, event: MouseEvent):
        if event.button == 'up':
            self._increment_t_index()
        elif event.button == 'down':
            self._decrement_t_index()
        self._fig.canvas.draw()

    def _update_keypress(self, event: KeyEvent):
        if event.key == 'alt+left':
            self._decrement_x_index()
        elif event.key == 'alt+right':
            self._increment_x_index()
        elif event.key == 'alt+up':
            self._increment_y_index()
        elif event.key == 'alt+down':
            self._decrement_y_index()
        elif event.key == 'ctrl+left':
            self._decrement_t_index()
        elif event.key == 'ctrl+right':
            self._increment_t_index()
        elif event.key == 'ctrl+up':
            self._increment_z_index()
        elif event.key == 'ctrl+down':
            self._decrement_z_index()
        elif event.key == '+':
            self._increment_vmax()
        elif event.key == '-':
            self._decrement_vmax()
        elif event.key == 'ctrl++':
            self._increment_vmin()
        elif event.key == 'ctrl+-':
            self._decrement_vmin()
        elif event.key == 'i':
            self._change_integrate_status()
        self._fig.canvas.draw()

    @property
    def _len(self):
        return self._data.__len__()

    @property
    def shape(self):
        return self._data.shape

    @property
    def _wcs(self):
        return self._wcs_list[self.t_index]

    @property
    def _wcs_xy(self):
        return self._wcs.dropaxis(0)

    @property
    def _wcs_xz(self):
        w = self._wcs.dropaxis(2)
        return w.swapaxes(0, 1)

    @property
    def _wcs_yz(self):
        return self._wcs.dropaxis(1)

    @property
    def _wcs_z(self):
        return self._wcs.dropaxis(1).dropaxis(1)

    @property
    def _pos(self):
        return self._wcs.pixel_to_world(self._z_index, self._x_index, self._y_index)

    @property
    def _x_pos(self):
        return self._pos[~1]

    @property
    def _y_pos(self):
        return self._pos[~0]

    @property
    def _z_pos(self):
        return self._pos[0]

    @property
    def _x_index_int(self):
        return int(round(self._x_index))

    @property
    def _y_index_int(self):
        return int(round(self._y_index))

    @property
    def _z_index_int(self):
        return int(round(self._z_index))

    @property
    def _lim_increment(self):
        return (self.vmax - self.vmin) / 10

    @property
    def vmin(self) -> float:
        return self._vmin

    @vmin.setter
    def vmin(self, value: float):
        self._vmin = value
        self._img_xy.set_clim(vmin=value)
        self._img_xz.set_clim(vmin=value)
        self._img_yz.set_clim(vmin=value)
        self._ax_z.set_ylim((value, self.vmax))

    @property
    def vmax(self) -> float:
        return self._vmax

    @vmax.setter
    def vmax(self, value: float):
        self._vmax = value
        if self._integrate:
            self._img_xy.set_clim(vmax=value * self._data.shape[-1])
            self._img_xz.set_clim(vmax=value)
            self._img_yz.set_clim(vmax=value)
            self._ax_z.set_ylim((self.vmin, value))
        else:
            self._img_xy.set_clim(vmax=value)
            self._img_xz.set_clim(vmax=value)
            self._img_yz.set_clim(vmax=value)
            self._ax_z.set_ylim((self.vmin, value))

    @property
    def _data_xy(self):
        if self._integrate:
            return np.sum(self._data[self.t_index, :, :, :], axis=-1)
        else:
            return self._data[self.t_index, :, :, self._z_index_int]

    @property
    def _data_xz(self):
        return self._data[self.t_index, self._y_index_int, :, :].T

    @property
    def _data_yz(self):
        return self._data[self.t_index, :, self._x_index_int, :]

    @property
    def _data_z(self):
        return self._data[self.t_index, self._y_index_int, self._x_index_int]

    @property
    def _data_z_extent(self):
        data_z_shp = self._data_z.shape
        start = np.array(0)
        end = np.array(data_z_shp[0])
        extent = self._wcs_z.all_pix2world((start[None, ...], end[None, ...]), 0)
        return np.squeeze(extent)

    @property
    def t_index(self) -> int:
        return self._t_index

    @t_index.setter
    def t_index(self, value: int):

        self._t_index = value

        self._fig.suptitle('t = ' + str(self.t_index))
        if self._integrate:
            self._ax_xy.set_title('Summed Z')
        else:
            self._ax_xy.set_title('z = ' + fmt.quantity(self._z_pos))
        self._ax_xy.reset_wcs(wcs = self._wcs_xy)
        self._img_xy.set_data(self._data_xy)
        self._xy_vline.set_xdata(self._x_index_int)
        self._xy_hline.set_ydata(self._y_index_int)

        self._ax_xz.set_title('y = ' + fmt.quantity(self._y_pos))
        self._ax_xz.reset_wcs(wcs=self._wcs_xz)
        self._img_xz.set_data(self._data_xz)
        self._xz_vline.set_xdata(self._x_index_int)
        self._xz_hline.set_ydata(self._z_index_int)

        self._ax_yz.set_title('x = ' + fmt.quantity(self._x_pos))
        self._ax_yz.reset_wcs(wcs=self._wcs_yz)
        self._img_yz.set_data(self._data_yz)
        self._yz_vline.set_xdata(self._z_index_int)
        self._yz_hline.set_ydata(self._y_index_int)

        self._ax_z.set_title('x = ' + fmt.quantity(self._x_pos) + ', y = ' + fmt.quantity(self._y_pos))
        self._ax_z.reset_wcs(wcs=self._wcs_z)
        self._plot_z[0].set_ydata(self._data_z)
        self._plot_z_med[0].set_ydata(self._data_xy_median[self.t_index])
        self._ax_z.set_ylim((self._vmin, self._vmax))
        self._z_vline.set_xdata(self._z_index)

    @property
    def x_index(self) -> float:
        return self._x_index

    @x_index.setter
    def x_index(self, value: float):
        self._x_index = value
        self._ax_yz.set_title('x = ' + fmt.quantity(self._x_pos))
        self._ax_z.set_title('x = ' + fmt.quantity(self._x_pos) +
                             ', y = ' + fmt.quantity(self._y_pos))
        self._xy_vline.set_xdata(self._x_index_int)
        self._xz_vline.set_xdata(self._x_index_int)
        self._img_yz.set_data(self._data_yz)
        self._plot_z[0].set_ydata(self._data_z)

    @property
    def y_index(self) -> float:
        return self._y_index

    @y_index.setter
    def y_index(self, value: float):
        self._y_index = value
        self._ax_xz.set_title('y = ' + fmt.quantity(self._y_pos))
        self._ax_z.set_title('x = ' + fmt.quantity(self._x_pos) +
                             ', y = ' + fmt.quantity(self._y_pos))
        self._xy_hline.set_ydata(self._y_index_int)
        self._yz_hline.set_ydata(self._y_index_int)
        self._img_xz.set_data(self._data_xz)
        self._plot_z[0].set_ydata(self._data_z)

    @property
    def z_index(self) -> float:
        return self._z_index

    @z_index.setter
    def z_index(self, value: float):
        self._z_index = value
        if self._integrate:
            self._ax_xy.set_title('Summed Z')
        else:
            self._ax_xy.set_title('z = ' + fmt.quantity(self._z_pos))
        self._xz_hline.set_ydata(self._z_index_int)
        self._yz_vline.set_xdata(self._z_index_int)
        self._z_vline.set_xdata(self._z_index_int)
        self._img_xy.set_data(self._data_xy)

    def _change_integrate_status(self):
        if self._integrate:
            self._integrate = False
            self._img_xy.set_data(self._data_xy)
            self._ax_xy.set_title('z = ' + fmt.quantity(self._z_pos))
            self._img_xy.set_clim(vmax=self.vmax)
        else:
            self._integrate = True
            self._img_xy.set_data(self._data_xy)
            self._ax_xy.set_title('Summed Z')
            self._img_xy.set_clim(vmax=self.vmax * self._data.shape[-1])



    def _increment_vmax(self):
        self.vmax = self.vmax + self._lim_increment

    def _decrement_vmax(self):
        self.vmax = self.vmax - self._lim_increment

    def _increment_vmin(self):
        self.vmin = self.vmin + self._lim_increment

    def _decrement_vmin(self):
        self.vmin = self.vmin - self._lim_increment

    def _increment_t_index(self):
        self.t_index = (self.t_index + 1) % self.shape[self._axis.t]

    def _decrement_t_index(self):
        self.t_index = (self.t_index - 1) % self.shape[self._axis.t]

    def _increment_x_index(self):
        self.x_index = (self.x_index + 1) % self.shape[self._axis.x]

    def _decrement_x_index(self):
        self.x_index = (self.x_index - 1) % self.shape[self._axis.x]

    def _increment_y_index(self):
        self.y_index = (self.y_index + 1) % self.shape[self._axis.y]

    def _decrement_y_index(self):
        self.y_index = (self.y_index - 1) % self.shape[self._axis.y]

    def _increment_z_index(self):
        self.z_index = (self.z_index + 1) % self.shape[self._axis.z]

    def _decrement_z_index(self):
        self.z_index = (self.z_index - 1) % self.shape[self._axis.z]

class TestHypercube:

    def test__init__(self, capsys):
        from kgpy.observatories.iris import mosaics
        import astropy.io.fits

        with capsys.disabled():
            mosaic_paths = mosaics.download()

            num_mosiacs = 2

            cube_list = []
            wcs_list = []
            for path in mosaic_paths[:num_mosiacs]:
                hdu = astropy.io.fits.open(path)[0]
                cube_list.append(np.moveaxis(hdu.data, 0, ~0))
                wcs = astropy.wcs.WCS(hdu.header)
                wcs = wcs.swapaxes(2, 1)
                wcs = wcs.swapaxes(1, 0)
                wcs_list.append(wcs)

            hypercube = np.array(cube_list)

            s = HypercubeSlicer(
                data=hypercube,
                wcs_list=wcs_list,
                width_ratios=(5, 1),
                height_ratios=(5, 1),
            )
            plt.show()
