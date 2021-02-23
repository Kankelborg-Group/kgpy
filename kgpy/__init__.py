"""
kgpy root package
"""
import typing as typ
import dataclasses
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import pandas
from kgpy import vector

__all__ = [
    'linspace', 'midspace',
    'Name',
    'fft',
    'rebin',
    'Trajectory'
]


@dataclasses.dataclass
class Name:
    """
    Representation of a hierarchical namespace.
    Names are a composition of a parent, which is also a name, and a base which is a simple string.
    The string representation of a name is <parent>.base, where <parent> is the parent's string expansion.
    """

    base: str = ''  #: Base of the name, this string will appear last in the string representation
    parent: 'typ.Optional[Name]' = None     #: Parent string of the name, this itself also a name

    def copy(self):
        if self.parent is not None:
            parent = self.parent.copy()
        else:
            parent = self.parent
        return type(self)(
            base=self.base,
            parent=parent,
        )

    def __add__(self, other: str) -> 'Name':
        """
        Quickly create the name of a child's name by adding a string to the current instance.
        Adding a string to a name instance returns
        :param other: A string representing the basename of the new Name instance.
        :return: A new `kgpy.Name` instance with the `self` as the `parent` and `other` as the `base`.
        """
        return type(self)(base=other, parent=self)

    def __repr__(self):
        if self.parent is not None:
            return self.parent.__repr__() + '.' + self.base

        else:
            return self.base


def rebin(arr: np.ndarray, scale_dims: typ.Tuple[int, ...]) -> np.ndarray:
    """
    Increases the size of an array by scale_dims in each i dimension by repeating each value scale_dims[i] times along
    that axis.

    :param arr: Array to modify
    :param scale_dims: Tuple with length ``arr.ndim`` specifying the size increase in each axis.
    :return: The resized array
    """
    new_arr = np.broadcast_to(arr, scale_dims + arr.shape)
    start_axes = np.arange(arr.ndim)
    new_axes = 2 * start_axes + 1
    new_arr = np.moveaxis(new_arr, start_axes, new_axes)

    new_shape = np.array(arr.shape) * np.array(scale_dims)
    new_arr = np.reshape(new_arr, new_shape)
    return new_arr


def linspace(start: np.ndarray, stop: np.ndarray, num: int, axis: int = 0) -> np.ndarray:
    """
    A modified version of :func:`numpy.linspace()` that returns a value in the center of the range between `start`
    and `stop` if `num == 1` unlike :func:`numpy.linspace` which would just return `start`.
    This function is often helfpul when creating a grid.
    Sometimes you want to test with only a single element, but you want that element to be in the center of the range
    and not off to one side.

    :param start: The starting value of the sequence.
    :param stop: The end value of the sequence, must be broadcastable with `start`.
    :param num: Number of samples to generate for this sequence.
    :param axis: The axis in the result used to store the samples. The default is the first axis.
    :return: An array the size of the broadcasted shape of `start` and `stop` with an additional dimension of length
        `num`.
    """
    if num == 1:
        return np.expand_dims((start + stop) / 2, axis=axis)
    else:
        return np.linspace(start=start, stop=stop, num=num, axis=axis)


def midspace(start: np.ndarray, stop: np.ndarray, num: int, axis: int = 0) -> np.ndarray:
    """
    A modified version of :func:`numpy.linspace` that selects cell centers instead of cell edges.

    :param start:
    :param stop:
    :param num:
    :param axis:
    :return:
    """
    a = np.linspace(start=start, stop=stop, num=num + 1, axis=axis)
    i0 = [slice(None)] * a.ndim
    i1 = i0.copy()
    i0[axis] = slice(None, ~0)
    i1[axis] = slice(1, None)
    return (a[i0] + a[i1]) / 2


@dataclasses.dataclass
class Trajectory:

    time: u.Quantity
    altitude: u.Quantity
    latitude: u.Quantity
    longitude: u.Quantity
    velocity: vector.Vector3D

    @classmethod
    def from_nsroc_csv(
            cls,
            csv_file: pathlib.Path,
            time_col: int = 1,
            altitude_col: int = 9,
            latitude_col: int = 10,
            longitude_col: int = 11,
            velocity_ew_col: int = 13,
            velocity_ns_col: int = 14,
            velocity_alt_col: int = 15,
    ):
        df = pandas.read_csv(
            csv_file,
            sep=' ',
            skipinitialspace=True,
            header=None,
            skiprows=1,
        )
        return cls(
            time=df[time_col].values * u.s,
            altitude=(df[altitude_col].values * u.m).to(u.km),
            latitude=df[latitude_col].values * u.deg,
            longitude=df[longitude_col].values * u.deg,
            velocity=vector.Vector3D(
                x=(df[velocity_ew_col].values * (u.m / u.s)).to(u.km / u.s),
                y=(df[velocity_ns_col].values * (u.m / u.s)).to(u.km / u.s),
                z=(df[velocity_alt_col].values * (u.m / u.s)).to(u.km / u.s),
            )
        )

    def plot_quantity_vs_time(
            self,
            quantity: u.Quantity,
            quantity_name: str = '',
            ax: typ.Optional[plt.Axes] = None,
    ):
        if ax is None:
            _, ax = plt.subplots()

        with astropy.visualization.quantity_support():
            ax.plot(
                self.time,
                quantity,
                label=quantity_name
            )

        ax.legend()

        return ax

    def plot_altitude_vs_time(
            self,
            ax: typ.Optional[plt.Axes] = None,
    ) -> plt.Axes:
        return self.plot_quantity_vs_time(
            quantity=self.altitude,
            quantity_name='altitude',
            ax=ax
        )

    def plot_total_velocity_vs_time(
            self,
            ax: typ.Optional[plt.Axes] = None,
    ) -> plt.Axes:
        return self.plot_quantity_vs_time(
            quantity=self.velocity.length,
            quantity_name='velocity',
            ax=ax
        )

    def plot_altitude_and_velocity_vs_time(
            self,
            ax_altitude: typ.Optional[plt.Axes] = None,
            ax_velocity: typ.Optional[plt.Axes] = None,
    ) -> typ.Tuple[plt.Axes, plt.Axes]:
        if ax_altitude is None:
            _, ax = plt.subplots()

        if ax_velocity is None:
            ax_velocity = ax_altitude.twinx()

        ax_altitude = self.plot_altitude_vs_time(ax=ax_altitude)
        ax_velocity.plot([], [])
        ax_velocity = self.plot_total_velocity_vs_time(ax=ax_velocity)

        ax_altitude.get_legend().remove()
        ax_velocity.get_legend().remove()

        ax_altitude.figure.legend(
            loc='upper right',
            bbox_to_anchor=(1, 1),
            bbox_transform=ax_altitude.transAxes,
        )

        return ax_altitude, ax_velocity
