
from win32com.client import constants

__all__ = ['ZemaxApertureType']


class ZemaxApertureTypeBase:

    @property
    def EntrancePuilDiameter(self) -> 'ZemaxApertureType':
        return constants.ZemaxApertureType_EntrancePupilDiameter

    @property
    def ImageSpaceFNum(self) -> 'ZemaxApertureType':
        return constants.ZemaxApertureType_ImageSpaceFNum

    @property
    def ObjectSpaceNA(self) -> 'ZemaxApertureType':
        return constants.ZemaxApertureType_ObjectSpaceNA

    @property
    def FloatByStopSize(self) -> 'ZemaxApertureType':
        return constants.ZemaxApertureType_FloatByStopSize

    @property
    def ParaxialWorkingFNum(self) -> 'ZemaxApertureType':
        return constants.ZemaxApertureType_ParaxialWorkingFNum

    @property
    def ObjectConeAngle(self) -> 'ZemaxApertureType':
        return constants.ZemaxApertureType_ObjectConeAngle


ZemaxApertureType = ZemaxApertureTypeBase()
