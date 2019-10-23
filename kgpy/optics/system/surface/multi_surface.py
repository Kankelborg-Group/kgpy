
import typing as tp
import astropy.units as u

from kgpy import math
from . import Aperture, Surface, CoordinateBreak

__all__ = ['MultiSurface']


class MultiSurface:



    def __init__(self, name: str):

        self._pre_tilt_decenter = CoordinateBreak(name + '.pre_coordinate_break')
        self._aperture = Surface(name + '.aperture')
        self._mechanical_aperture = Surface(name + '.mechanical_aperture')
        self._post_tilt_decenter = CoordinateBreak(name + '.post_coordinate_break')
        self._thickness = Surface(name + '.post_coordinate_break')

    @property
    def aperture(self) -> Aperture:
        return self._aperture.aperture

    @aperture.setter
    def aperture(self, value: Aperture):
        self.aperture.aperture = value

    @property
    def mechanical_aperture(self) -> Aperture:
        return self._mechanical_aperture.aperture

    @mechanical_aperture.setter
    def mechanical_aperture(self, value: Aperture):
        self._mechanical_aperture.aperture = value
        
    @property
    def pre_tilt_decenter(self) -> math.geometry.CoordinateSystem:
        return self._pre_tilt_decenter.pre_tilt_decenter
    
    @pre_tilt_decenter.setter
    def pre_tilt_decenter(self, value: math.CoordinateSystem):
        self._pre_tilt_decenter.pre_tilt_decenter = value

    @property
    def post_tilt_decenter(self) -> math.CoordinateSystem:
        return self._post_tilt_decenter.post_tilt_decenter

    @post_tilt_decenter.setter
    def post_tilt_decenter(self, value: math.CoordinateSystem):
        self._post_tilt_decenter.post_tilt_decenter = value

    @property
    def thickness(self) -> u.Quantity:
        return self._thickness.thickness

    @thickness.setter
    def thickness(self, value: u.Quantity):
        self._thickness.thickness = value
