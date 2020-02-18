from . import Operand

__all__ = ['SurfaceOperand']


class SurfaceOperand(Operand):

    @property
    def surface_index(self) -> int:
        return self.param_1

    @surface_index.setter
    def surface_index(self, value: int):
        self.param_1 = value
