
import numpy as np
import astropy.units as u

from kgpy import math, optics

from . import Surface, Material, Aperture

__all__ = ['Standard']


class Standard(Surface):

    def __init__(self,
                 *args,
                 radius: u.Quantity = None,
                 material: Material = None,
                 conic: u.Quantity = None,
                 aperture: Aperture = None,
                 pre_tilt_decenter: math.geometry.CoordinateSystem = None,
                 post_tilt_decenter: math.geometry.CoordinateSystem = None,
                 **kwargs
                 ):

        super().__init__(*args, **kwargs)

        if radius is None:
            radius = np.inf * u.m
        if conic is None:
            conic = 0 * u.dimensionless_unscaled
        if pre_tilt_decenter is None:
            pre_tilt_decenter = math.geometry.CoordinateSystem()
        if post_tilt_decenter is None:
            post_tilt_decenter = math.geometry.CoordinateSystem()

        self._radius = radius
        self._material = material
        self._conic = conic
        self._aperture = aperture
        self._pre_tilt_decenter = pre_tilt_decenter
        self._post_tilt_decenter = post_tilt_decenter


    @property
    def name(self):
        return self._name

    @property
    def thickness(self):
        return self._thickness

    @property
    def radius(self):
        return self._radius

    @property
    def material(self) -> Material:
        return self._material

    @property
    def conic(self) -> u.Quantity:
        return self._conic

    @property
    def aperture(self) -> Aperture:
        return self._aperture

    @property
    def pre_tilt_decenter(self) -> math.geometry.CoordinateSystem:
        return self._pre_tilt_decenter

    @property
    def post_tilt_decenter(self) -> math.geometry.CoordinateSystem:
        return self._post_tilt_decenter



