
import collections
import typing as tp

from kgpy import optics

from . import Operation

__all__ = ['OperationList']


class OperationList(collections.UserList):

    def __init__(self, operations: tp.List[Operation] = None):

        super().__init__(operations)

        self.configuration = None      # type: tp.Optional[optics.zemax.system.Configuration]
