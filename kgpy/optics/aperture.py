
from typing import Tuple
from shapely.geometry import Polygon


class Aperture:

    def __init__(self, decenter: Tuple[float, float], is_obscuration=False):

        self.is_obscuration = is_obscuration

        self.decenter_X = decenter[0]
        self.decenter_Y = decenter[1]


class RectangularAperture(Aperture):

    def __init__(self, half_width: Tuple[float, float], decenter: Tuple[float, float], is_obscuration=False):

        super().__init__(decenter, is_obscuration)

        self.half_width_X = half_width[0]
        self.half_width_Y = half_width[1]


class CircularAperture(Aperture):

    def __init__(self, radius: Tuple[float, float], decenter: Tuple[float, float], is_obscuration=False):

        super().__init__(decenter, is_obscuration)

        self.min_radius = radius[1]
        self.max_radius = radius[0]


class UserAperture(Aperture):
    
    def __init__(self, shape: Polygon, decenter: Tuple[float, float], is_obscuration=False):

        super().__init__(decenter, is_obscuration)

        self.shape = shape
