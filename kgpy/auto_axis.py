import typing as typ
import abc

__all__ = ['AutoAxis']


class AutoAxis:

    @abc.abstractmethod
    def __init__(self):
        super().__init__()
        self.ndim = 0
        self.all = []

    def auto_axis_index(self):
        i = ~self.ndim
        self.all.append(i)
        self.ndim += 1
        return i

    def perp_axes(self, axis: int) -> typ.Tuple[int, ...]:
        axes = self.all.copy()
        axes = [a % self.ndim for a in axes]
        axes.remove(axis % self.ndim)
        return tuple([a - self.ndim for a in axes])
