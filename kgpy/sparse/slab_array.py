import numpy as np
import typing as typ
import dataclasses

__all__  = ['SlabArray']


@dataclasses.dataclass
class SlabArray:
    data: typ.List[np.ndarray]

    def __iter__(self):
        return self.data.__iter__()

    def __len__(self):
        return self.data.__len__()

    def copy(self):
        new_data = []
        for d in self:
            new_data.append(d.copy())
        return type(self)(new_data)

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
            for d, f in zip(self, other):
                new_d = d.copy().__truediv__(f.copy())
                new_data.append(new_d)

        else:
            raise ValueError('Slab Array lengths don\'t match')

        return type(self)(new_data)

