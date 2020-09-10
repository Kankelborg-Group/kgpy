import dataclasses
import numpy as np
from kgpy import mixin

__all__ = ['Obscurable']


@dataclasses.dataclass
class Obscurable(
    mixin.Copyable,
):
    is_obscuration: bool = False

    def copy(self) -> 'Obscurable':
        other = super().copy()      # type: Obscurable
        other.is_obscuration = self.is_obscuration
        return other
