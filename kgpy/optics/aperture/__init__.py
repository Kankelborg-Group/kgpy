"""
Package to represent the aperture of optical surfaces.
"""

__all__ = ['Aperture', 'Decenterable', 'Obscurable', 'Circular', 'Polygon', 'Rectangular', 'RegularPolygon',
           'IrregularPolygon',
           'IsoscelesTrapezoid', 'AsymmetricRectangular']

from ._aperture import Aperture
from ._decenterable import Decenterable
from ._obscurable import Obscurable
from ._circular import Circular
from ._polygon import Polygon
from ._rectangular import Rectangular
from ._regular_polygon import RegularPolygon
from ._irregular_polygon import IrregularPolygon
from ._general_polygon import GeneralPolygon
from ._isosceles_trapezoid import IsoscelesTrapezoid
from ._asymmetric_rectangular import AsymmetricRectangular

Aperture.__module__ = __name__
Decenterable.__module__ = __name__
Obscurable.__module__ = __name__
Circular.__module__ = __name__
Polygon.__module__ = __name__
Rectangular.__module__ = __name__
RegularPolygon.__module__ = __name__
IrregularPolygon.__module__ = __name__
GeneralPolygon.__module__ = __name__
IsoscelesTrapezoid.__module__ = __name__
AsymmetricRectangular.__module__ = __name__
