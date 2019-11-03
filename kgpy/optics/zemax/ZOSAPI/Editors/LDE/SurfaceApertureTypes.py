
from win32com.client import constants

__all__ = ['SurfaceApertureTypes']


class SurfaceApertureTypesBase:

    # @property
    # def (self) -> 'SurfaceApertureTypes':
    #     return constants.

    # CircularAperture = constants.SurfaceApertureTypes_CircularAperture              # type: SurfaceApertureTypes
    # CircularObscuration = constants.SurfaceApertureTypes_CircularObscuration        # type: SurfaceApertureTypes
    # Spider = constants.SurfaceApertureTypes_Spider                                  # type: SurfaceApertureTypes
    # RectangularAperture = constants.SurfaceApertureTypes_RectangularAperture        # type: SurfaceApertureTypes
    # RectangularObscuration = constants.SurfaceApertureTypes_RectangularObscuration  # type: SurfaceApertureTypes
    # EllipticalAperture = constants.SurfaceApertureTypes_EllipticalAperture          # type: SurfaceApertureTypes
    # EllipticalObscuration = constants.SurfaceApertureTypes_EllipticalObscuration    # type: SurfaceApertureTypes
    # UserAperture = constants.SurfaceApertureTypes_UserAperture                      # type: SurfaceApertureTypes
    # UserObscuration = constants.SurfaceApertureTypes_UserObscuration                # type: SurfaceApertureTypes
    # FloatingAperture = constants.SurfaceApertureTypes_FloatingAperture              # type: SurfaceApertureTypes

    @property
    def none(self) -> 'SurfaceApertureTypes':
        return constants.SurfaceApertureTypes_None

    @property
    def CircularAperture(self) -> 'SurfaceApertureTypes':
        return constants.SurfaceApertureTypes_CircularAperture

    @property
    def CircularObscuration(self) -> 'SurfaceApertureTypes':
        return constants.SurfaceApertureTypes_CircularObscuration

    @property
    def Spider(self) -> 'SurfaceApertureTypes':
        return constants.SurfaceApertureTypes_Spider

    @property
    def RectangularAperture(self) -> 'SurfaceApertureTypes':
        return constants.SurfaceApertureTypes_RectangularAperture

    @property
    def RectangularObscuration(self) -> 'SurfaceApertureTypes':
        return constants.SurfaceApertureTypes_RectangularObscuration

    @property
    def EllipticalAperture(self) -> 'SurfaceApertureTypes':
        return constants.SurfaceApertureTypes_EllipticalAperture

    @property
    def EllipticalObscuration(self) -> 'SurfaceApertureTypes':
        return constants.SurfaceApertureTypes_EllipticalObscuration

    @property
    def UserAperture(self) -> 'SurfaceApertureTypes':
        return constants.SurfaceApertureTypes_UserAperture

    @property
    def UserObscuration(self) -> 'SurfaceApertureTypes':
        return constants.SurfaceApertureTypes_UserObscuration

    @property
    def FloatingAperture(self) -> 'SurfaceApertureTypes':
        return constants.SurfaceApertureTypes_FloatingAperture


# Make singleton class
SurfaceApertureTypes = SurfaceApertureTypesBase()

