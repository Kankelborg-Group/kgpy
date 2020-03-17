import numpy as np
import typing as typ
import dataclasses

__all__ = ['SlabArray']


@dataclasses.dataclass
class SlabArray:
    data: typ.List[np.ndarray]
    start_indices: 'np.ndarray[int]'
    slab_axis: int = ~0

    @property
    def ndim(self):
        return self.data[0].ndim

    def __iter__(self):
        return self.data.__iter__()

    def __len__(self):
        return self.data.__len__()

    def copy(self):
        new_data = []
        for d in self:
            new_data.append(d.copy())
        return type(self)(new_data, self.start_indices)

    def __neg__(self) -> 'SlabArray':
        new_data = []
        for d in self:
            new_d = d.copy().__neg__()
            new_data.append(new_d)
        return type(self)(new_data)

    def __add__(self, other: typ.Union[float, 'SlabArray']) -> 'SlabArray':
        new_data = []

        if np.isscalar(other):
            for d in self:
                new_d = d.copy().__add__(other)
                new_data.append(new_d)

        elif len(self) == len(other):
            self._check_compatibility(other)
            for d, f in zip(self, other):
                new_d = d.copy().__add__(f.copy())
                new_data.append(new_d)

        else:
            raise ValueError('Slab Array lengths don\'t match')

        return type(self)(new_data)

    def __sub__(self, other: typ.Union[float, 'SlabArray']) -> 'SlabArray':
        return self.__add__(-other)

    def __mul__(self, other: typ.Union[float, 'SlabArray']) -> 'SlabArray':
        new_data = []

        if np.isscalar(other):
            for d in self:
                new_d = d.copy().__mul__(other)
                new_data.append(new_d)

        elif len(self) == len(other):
            self._check_compatibility(other)
            for d, f in zip(self, other):
                new_d = d.copy().__mul__(f.copy())
                new_data.append(new_d)

        else:
            raise ValueError('Slab Array lengths don\'t match')

        return type(self)(new_data)

    def __truediv__(self, other: typ.Union[float, 'SlabArray']) -> 'SlabArray':
        new_data = []

        if np.isscalar(other):
            for d in self:
                new_d = d.copy().__truediv__(other)
                new_data.append(new_d)

        elif len(self) == len(other):
            self._check_compatibility(other)
            for d, f in zip(self, other):
                new_d = d.copy().__truediv__(f.copy())
                new_data.append(new_d)

        else:
            raise ValueError('Slab Array lengths don\'t match')

        return type(self)(new_data)

    def _check_compatibility(self, other):
        if not np.all(self.start_indices == other.start_indices):
            raise ValueError('Incompatible reference indices.')
        if self.slab_axis != other.slab_axis:
            raise ValueError('Both SlabArrays must have the same slab axis')

    def sum(self, axis: typ.Union[int, typ.Tuple[int, ...]] = None):
        if axis is None:
            axis = tuple(np.arange(self.ndim))

        if np.isscalar(axis):
            axis = (axis,)

        new_data = []
        for d in self:
            new_d = d.sum(axis, keepdims=True)
            new_data.append(new_d)

        if self.slab_axis in axis:
            fin_d = np.zeros_like(new_data[0])
            for d in new_data:
                fin_d += d
            final_data = [fin_d]

            final_start_indices = np.array([0])

        else:
            final_data = new_data
            final_start_indices = self.start_indices.copy()

        return type(self)(final_data, final_start_indices, self.slab_axis)

    def __pow__(self, power, modulo=None):
        new_data = []
        for sl in self.data:
            new_slab = np.float_power(sl, power)
            new_data.append(new_slab)

        return type(self)(new_data, self.start_indices, self.slab_axis)



class SlabArray2(np.ndarray):

    def __array_finalize__(self, obj):
        # super(SlabArray2, self).__array_finalize__(self, obj)
        pass


