import dataclasses
import typing as typ
import numpy as np
import nptyping

X = typ.TypeVar('X')
S = typ.TypeVar('S')

ItemBase = typ.Union[int, slice]
Item = typ.Union[ItemBase, typ.Tuple[ItemBase, ...]]


class Array(typ.Generic[X]):

    Y = typ.Union[X, 'Array[X]']

    def __getitem__(self, item: Item) -> Y:
        pass

    def __setitem__(self, key: Item, value: Y):
        pass

    def __iter__(self) -> typ.Iterator[Y]:
        pass
    
    def __add__(self, other: Y) -> 'Array[X]':
        pass
