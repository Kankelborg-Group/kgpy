import typing as typ
import abc
import collections
import dataclasses
import copy

import numpy
import numpy as np
import astropy.units as u
import kgpy
import kgpy.mixin

__all__ = [
    'Linear'
]


@dataclasses.dataclass
class Interpolator(
    kgpy.mixin.Copyable,
    abc.ABC,
):
    data: u.Quantity
    grid: typ.Sequence[u.Quantity]
    # axis: typ.Optional[typ.Union[int, typ.Sequence[int]]] = None

    @property
    def shape(self):
        return np.broadcast(self.data, *self.grid).shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def data_final(self) -> u.Quantity:
        return np.broadcast_to(self.data, self.shape, subok=True)

    @property
    def grid_final(self) -> typ.List[u.Quantity]:
        shape = self.shape
        return [np.broadcast_to(x, shape=shape, subok=True) for x in self.grid]

    # @property
    # def axis_normalized(self) -> typ.Tuple[int]:
    #     if self.axis is None:
    #         axis = tuple(range(self.data.ndim))
    #     elif not isinstance(self.axis, collections.Sequence):
    #         axis = self.axis,
    #     else:
    #         axis = tuple(self.axis)
    #     return axis

    @abc.abstractmethod
    def __call__(self, grid: typ.Sequence[u.Quantity]):
        pass

    def view(self) -> 'Interpolator':
        other = super().view()  # type: Interpolator
        other.data = self.data
        other.grid = self.grid
        # other.axis = self.axis
        return other

    def copy(self) -> 'Interpolator':
        other = self.copy()     # type: Interpolator
        other.data = self.data.copy()
        other.grid = copy.deepcopy(self.grid)
        # other.axis = copy.deepcopy(self.axis)
        return other


@dataclasses.dataclass
class NearestNeighbor(Interpolator):

    def calc_index_nearest(self, grid: typ.Sequence[u.Quantity]) -> typ.Tuple[np.ndarray, ...]:

        shape_grid = np.broadcast(*grid).shape

        distance_squared = np.zeros(shape_grid + self.shape)

        for i in range(len(grid)):
            distance_squared = distance_squared + np.square(grid[i][(..., ) + self.ndim * (np.newaxis, )] - self.grid[i])

        index = np.argmin(distance_squared.reshape(shape_grid + (-1,)), axis=~0)
        index = np.unravel_index(index, self.shape)

        return index

    def __call__(self, grid: typ.Sequence[u.Quantity]):
        return self.data_final[self.calc_index_nearest(grid)]


@dataclasses.dataclass
class Linear(Interpolator):

    def calc_index_lower(self, grid: typ.Sequence[u.Quantity]) -> typ.Tuple[np.ndarray, ...]:

        shape_grid = np.broadcast(*grid).shape

        distance = np.zeros(shape_grid + self.shape)

        for i in range(len(grid)):
            distance_i = self.grid[i] - grid[i][(..., ) + self.ndim * (np.newaxis,)]
            distance_i[distance_i > 0] = -np.inf
            distance = distance + distance_i

        index = np.argmax(distance.reshape(shape_grid + (-1,)), axis=~0)
        index = np.unravel_index(index, self.shape)
        for i, ind in enumerate(index):
            ind_max = self.shape[i] - 2
            ind[ind > ind_max] = ind_max

        return index

    @classmethod
    def lerp(
            cls,
            x: u.Quantity,
            x0: u.Quantity,
            x1: u.Quantity,
            y0: u.Quantity,
            y1: u.Quantity,
    ) -> u.Quantity:
        # return (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    def eval_recursive(
            self,
            grid: typ.Sequence[u.Quantity],
            index_lower: typ.Optional[typ.Tuple[np.ndarray, ...]] = None,
            axis: int = 0,
    ):

        if index_lower is None:
            index_lower = self.calc_index_lower(grid)

        index_upper = list(index_lower)
        index_upper[axis] = index_upper[axis] + 1

        if axis + 1 < len(self.grid):
            axis_new = axis + 1
            y0 = self.eval_recursive(grid=grid, index_lower=index_lower, axis=axis_new)
            y1 = self.eval_recursive(grid=grid, index_lower=index_upper, axis=axis_new)

        else:
            data = self.data_final
            y0 = data[index_lower]
            y1 = data[index_upper]

        grid_data = self.grid_final

        return self.lerp(
            x=grid[axis],
            x0=grid_data[axis][index_lower],
            x1=grid_data[axis][index_upper],
            y0=y0,
            y1=y1,
        )

    def __call__(self, grid: typ.Sequence[u.Quantity]):
        return self.eval_recursive(
            grid=grid
        )



