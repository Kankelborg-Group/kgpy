
from win32com.client import constants

__all__ = ['ZemaxSystemUnits']


class ZemaxSystemUnits:

    @property
    def Millimeters(self) -> int:
        return constants.ZemaxSystemUnits_Millimeters

    @property
    def Centimeters(self) -> int:
        return constants.ZemaxSystemUnits_Centimeters

    @property
    def Inches(self) -> int:
        return constants.ZemaxSystemUnits_Inches

    @property
    def Meters(self) -> int:
        return constants.ZemaxSystemUnits_Meters


# Make class singleton class.
# This makes all the methods appear to be static
# This was suggested by the following stackoverflow answer: https://stackoverflow.com/a/12330859
ZemaxSystemUnits = ZemaxSystemUnits()
