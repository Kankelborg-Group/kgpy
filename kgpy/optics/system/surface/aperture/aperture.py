import dataclasses
import typing as typ
import numpy as np

from ... import mixin

__all__ = ['Aperture']


@dataclasses.dataclass
class Aperture(mixin.Broadcastable):

    pass
