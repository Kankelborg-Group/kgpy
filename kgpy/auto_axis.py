import typing as typ
import abc

__all__ = ['AutoAxis']


class AutoAxis:
    """
    Semi-automated axis numbering
    """

    @abc.abstractmethod
    def __init__(self):
        super().__init__()
        self.num_left_dim = 0
        self.num_right_dim = 0
        self.all = []

    @property
    def ndim(self) -> int:
        return self.num_left_dim + self.num_right_dim

    def auto_axis_index(self, from_right: bool = True):
        if from_right:
            i = ~self.num_right_dim
            self.num_right_dim += 1
        else:
            i = self.num_left_dim
            self.num_left_dim += 1
        self.all.append(i)
        return i

    def perp_axes(self, axis: int) -> typ.Tuple[int, ...]:
        axes = self.all.copy()
        axes = [a % self.ndim for a in axes]
        axes.remove(axis % self.ndim)
        return tuple([a - self.ndim for a in axes])
