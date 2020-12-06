__all__ = ['ISurfaceApertureType', 'ISurfaceApertureCircular', 'ISurfaceApertureElliptical', 'ISurfaceApertureFloating',
           'ISurfaceApertureNone', 'ISurfaceApertureRectangular', 'ISurfaceApertureSpider', 'ISurfaceApertureUser']


class ISurfaceApertureType:
    _S_CircularAperture = None  # type: ISurfaceApertureCircular
    _S_CircularObscuration = None  # type: ISurfaceApertureCircular
    _S_EllipticalAperture = None  # type: ISurfaceApertureElliptical
    _S_EllipticalObscuration = None  # type: ISurfaceApertureElliptical
    _S_FloatingAperture = None  # type: ISurfaceApertureFloating
    _S_None = None  # type: ISurfaceApertureNone
    _S_RectangularAperture = None  # type: ISurfaceApertureRectangular
    _S_RectangularObscuration = None  # type: ISurfaceApertureRectangular
    _S_Spider = None  # type: ISurfaceApertureSpider
    _S_UserAperture = None  # type: ISurfaceApertureUser
    _S_UserObscuration = None  # type: ISurfaceApertureUser
    IsReadOnly = None  # type: bool
    Type = None  # type: ISurfaceApertureType


class ISurfaceApertureCircular(ISurfaceApertureType):
    ApertureXDecenter = None  # type: float
    ApertureYDecenter = None  # type: float
    MaximumRadius = None  # type: float
    MinimumRadius = None  # type: float


class ISurfaceApertureElliptical(ISurfaceApertureType):
    ApertureXDecenter = None  # type: float
    ApertureYDecenter = None  # type: float
    XHalfWidth = None  # type: float
    YHalfWidth = None  # type: float


class ISurfaceApertureFloating(ISurfaceApertureType):
    pass


class ISurfaceApertureNone(ISurfaceApertureType):
    pass


class ISurfaceApertureRectangular(ISurfaceApertureType):
    ApertureXDecenter = None  # type: float
    ApertureYDecenter = None  # type: float
    XHalfWidth = None  # type: float
    YHalfWidth = None  # type: float


class ISurfaceApertureSpider(ISurfaceApertureType):
    ApertureXDecenter = None  # type: float
    ApertureYDecenter = None  # type: float
    NumberOfArms = None  # type: int
    WidthOfArms = None  # type: float


class ISurfaceApertureUser(ISurfaceApertureType):
    ApertureFile = None  # type: str
    ApertureXDecenter = None  # type: float
    ApertureYDecenter = None  # type: float
    UDAScale = None  # type: float
