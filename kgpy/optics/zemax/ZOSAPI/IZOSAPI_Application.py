
from kgpy.optics.zemax import ZOSAPI

__all__ = ['IZOSAPI_Application']


class IZOSAPI_Application:

    AutodeskInventorFilesDir = None             # type: str
    CoatingDir = None                           # type: str
    CreoParametricFilesDir = None               # type: str
    ExpirationDay = None                        # type: int
    ExpirationMonth = None                      # type: int
    ExpirationYear = None                       # type: int
    GlassDir = None                             # type: str
    ImagesDir = None                            # type: str
    IsValidLicenseForAPI = None                 # type: bool
    LensDir = None                              # type: str
    LicenseStatus = None                        # type: ZOSAPI.LicenseStatusType
    MATLABFilesDir = None                       # type: str
    Mode = None                                 # type: ZOSAPI.ZOSAPI_Mode
    NumberOfCPUs = None                         # type: int
    NumberOfOpticalSystems = None               # type: int
    ObjectsDir = None                           # type: str
    OperandArgument1 = None                     # type: float
    OperandArgument2 = None                     # type: float
    OperandArgument3 = None                     # type: float
    OperandArgument4 = None                     # type: float
    OperandResults = None                       # type: ZOSAPI.IVectorData
    OpticStudioVersion = None                   # type: int
    POPDir = None                               # type: str
    PrimarySystem = None                        # type: ZOSAPI.IOpticalSystem
    ProgramDir = None                           # type: str
    ProgressMessage = None                      # type: str
    ProgressPercent = None                      # type: float
    SamplesDir = None                           # type: str
    ScatterDir = None                           # type: str
    SerialCode = None                           # type: str
    ShowChangesInUI = None                      # type: bool
    SolidWorksFilesDir = None                   # type: str
    TerminateRequested = None                   # type: bool
    UndoDir = None                              # type: str
    UserAnalysisData = None                     # type: ZOSAPI.Analysis.IUserAnalysisData
    ZemaxDataDir = None                         # type: str
    ZPLDir = None                               # type: str

    def CloseApplication(self) -> None:
        pass

    def CloseSystemAt(self, pos: int, saveIfNeeded: bool) -> bool:
        pass

    def CreateNewSystem(self, mode: ZOSAPI.SystemType) -> ZOSAPI.IOpticalSystem:
        pass

    def GetDate(self) -> str:
        pass

    def GetSystemAt(self, pos: int) -> ZOSAPI.IOpticalSystem:
        pass

    def LoadNewSystem(self, LensFile: str) -> ZOSAPI.IOpticalSystem:
        pass

    def UpdateFileLists(self):
        pass