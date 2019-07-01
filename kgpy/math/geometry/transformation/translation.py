
import typing as tp

from kgpy import math

__all__ = ['Translation']


class Translation(math.geometry.Vector):

    def __init__(self):

        super().__init__()

        self.coordinate_system = None

    @property
    def coordinate_system(self) -> tp.Optional[math.geometry.CoordinateSystem]:
        return self._coordinate_system

    @coordinate_system.setter
    def coordinate_system(self, value: tp.Optional[math.geometry.CoordinateSystem]):
        self._coordinate_system = value

    def __neg__(self) -> 'Translation':
        super().__neg__()
