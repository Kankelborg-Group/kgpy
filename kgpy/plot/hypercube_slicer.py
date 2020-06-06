import typing as typ
import numpy as np
import matplotlib.pyplot as plt
import astropy.wcs

__all__ = []


class Axis:
    t = ~3
    x = ~2
    y = ~1
    z = ~0


class HypercubeSlicer:

    _i_xy = (0, 0)
    _i_xz = (1, 0)
    _i_yz = (0, 1)
    _i_z = (1, 1)

    _axis = Axis()
    
    def __init__(
            self,
            data: np.ndarray,
            wcs_list: typ.Optional[typ.List[astropy.wcs.WCS]] = None,
            start_index: int = 0,
    ):

        self._index = 0
        self._data = data
        self._wcs_list = wcs_list

        sh = data.shape
        gridspec_width = (sh[self._axis.x], sh[self._axis.z])
        gridspec_height = (sh[self._axis.y], sh[self._axis.w])

        fig, axs = plt.subplots(
            nrows=2,
            ncols=2,
            dpi=200,
            gridspec_kw={'width_ratios': gridspec_width, 'height_ratios': gridspec_height},
            constrained_layout=True,
        )

        self._fig = fig     # type: plt.Figure
        self._axs = axs

        fig.canvas.mpl_connect()

    @property
    def _len(self):
        return self._data.__len__()

    @property
    def _ax_xy(self) -> plt.Axes:
        return self._axs[self._i_xy]

    @_ax_xy.setter
    def _ax_xy(self, value: plt.Axes):
        self._axs[self._i_xy] = value

    @property
    def _ax_yz(self) -> plt.Axes:
        return self._axs[self._i_yz]

    @_ax_yz.setter
    def _ax_yz(self, value: plt.Axes):
        self._axs[self._i_yz] = value

    @property
    def _ax_xz(self) -> plt.Axes:
        return self._axs[self._i_xz]

    @_ax_xz.setter
    def _ax_xz(self, value: plt.Axes):
        self._axs[self._i_xz] = value

    @property
    def _ax_z(self) -> plt.Axes:
        return self._axs[self._i_z]

    @_ax_z.setter
    def _ax_z(self, value: plt.Axes):
        self._axs[self._i_z] = value

    @property
    def index(self) -> int:
        return self._index

    @index.setter
    def index(self, value: int):
        self._index = value

        if self._wcs_list is not None:
            wcs = self._wcs_list[self.index]
            wcs_xy = wcs.dropaxis(0)
            wcs_xz = wcs.dropaxis(1)
            wcs_yz = wcs.dropaxis(2)
            wcs_z = wcs_xz.dropaxis(1)
        else:
            wcs_xy = None
            wcs_xz = None
            wcs_yz = None
            wcs_z = None

        self._ax_xy = self._fig.add_subplot(self._ax_xy, projection=wcs_xy)



class TestHypercube:

    pass

