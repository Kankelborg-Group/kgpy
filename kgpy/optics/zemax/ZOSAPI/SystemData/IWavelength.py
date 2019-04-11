
__all__ = ['IWavelength']

class IWavelength:

    IsActive = None             # type: bool
    IsPrimary = None            # type: bool
    Wavelength = None           # type: float
    WavelengthNumber = None     # type: int
    Weight = None               # type: float

    def MakePrimary(self) -> None:
        pass