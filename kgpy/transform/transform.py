import typing as typ
import abc
import kgpy.mixin

__all__ = ['Transform']


class Transform(
    kgpy.mixin.Copyable,
    kgpy.mixin.Broadcastable,
    abc.ABC,
):
    pass
