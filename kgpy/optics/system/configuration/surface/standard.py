
from kgpy import math, optics

from . import Surface

__all__ = ['Standard']


class Standard(Surface):

    def __init__(self, name: str):

        super().__init__(name)

        self.pre_tilt_decenter = math.geometry.CoordinateSystem()
        self.post_tilt_decenter = math.geometry.CoordinateSystem()

    @property
    def configuration(self) -> optics.system.Configuration:
        return super().configuration

    @configuration.setter
    def configuration(self, value: optics.system.Configuration):
        super().configuration = value

        self.pre_cs = self.pre_cs_

    @property
    def pre_tilt_decenter(self) -> math.geometry.CoordinateSystem:
        return self._pre_tilt_decenter

    @pre_tilt_decenter.setter
    def pre_tilt_decenter(self, value: math.geometry.CoordinateSystem):
        self._pre_tilt_decenter = value

    @property
    def post_tilt_decenter(self) -> math.geometry.CoordinateSystem:
        return self._post_tilt_decenter

    @post_tilt_decenter.setter
    def post_tilt_decenter(self, value: math.geometry.CoordinateSystem):
        self._post_tilt_decenter = value

    @property
    def pre_cs(self) -> math.geometry.CoordinateSystem:
        return self._pre_cs

    @pre_cs.setter
    def pre_cs(self, value: math.geometry.CoordinateSystem):
        self._pre_cs = value

        self.front_cs = self.front_cs_

    @property
    def pre_cs_(self):

        try:
            return self.previous_surface.back_cs

        except AttributeError:
            return math.geometry.CoordinateSystem()

    @property
    def front_cs(self) -> math.geometry.CoordinateSystem:
        return self._front_cs

    @front_cs.setter
    def front_cs(self, value: math.geometry.CoordinateSystem):
        self._front_cs = value

        self.post_cs = self.post_cs_

    @property
    def front_cs_(self) -> math.geometry.CoordinateSystem:
        return self.pre_cs @ self.pre_tilt_decenter

    @property
    def post_cs(self) -> math.geometry.CoordinateSystem:
        return self._post_cs

    @post_cs.setter
    def post_cs(self, value: math.geometry.CoordinateSystem):
        self._post_cs = value

        self.back_cs = self.back_cs_

    @property
    def post_cs_(self):
        return self.cs @ self.post_tilt_decenter

    @property
    def back_cs(self) -> math.geometry.CoordinateSystem:
        return self._back_cs

    @back_cs.setter
    def back_cs(self, value: math.geometry.CoordinateSystem):
        self._back_cs = value

        try:
            self.next_surface.pre_cs = self.next_surface.pre_cs_

        except AttributeError:
            pass

    @property
    def back_cs_(self):
        return self.post_cs + self.T
