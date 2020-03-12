from . import Tilt, Decenter, TiltFirst
from .. import before

__all__ = ['TiltDecenter']


class TiltDecenter(before.TiltDecenter):

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

    @property
    def tilt_first(self) -> TiltFirst:
        return super().tilt_first

    @tilt_first.setter
    def tilt_first(self, value: TiltFirst):
        super().tilt_first = value
