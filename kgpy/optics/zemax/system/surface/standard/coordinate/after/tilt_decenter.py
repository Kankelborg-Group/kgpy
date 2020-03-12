from . import Tilt, Decenter
from .. import before

__all__ = ['TiltDecenter']


class TiltDecenter(before.tilt.TiltDecenter):

    @property
    def tilt(self) -> Tilt:
        return super().tilt

    @tilt.setter
    def tilt(self, value: Tilt):
        super().tilt = value

    @property
    def decenter(self) -> Decenter:
        return self._decenter

    @decenter.setter
    def decenter(self, value: Decenter):
        super().decenter = value