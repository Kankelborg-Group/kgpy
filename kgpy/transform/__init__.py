"""
Package for general coordinate transforms.
Currently, the only supported transforms are rigid transformations.
"""
import abc
from kgpy import mixin


class Transform(
    mixin.Toleranceable,
    mixin.Copyable,
    mixin.Broadcastable,
    abc.ABC,
):
    pass


from . import rigid

__all__ = [
    'Transform',
    'rigid',
]
