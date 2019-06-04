
import collections
import typing as t

from kgpy.instrument.spectrograph.spectrograph import Spectrograph

__all__ = ['Array']


class Array(collections.UserList):

    def __init__(self, spectrogaph_list: t.List[Spectrograph]):

        super().__init__(spectrogaph_list)
