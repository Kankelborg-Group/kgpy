
from kgpy.optics.zemax import ZOSAPI

__all__ = ['IA_']


class IA_:

    AnalysisType = None                     # type: ZOSAPI.Analysis.AnalysisIDM
    GetAnalysisName = None                  # type: str
    HasAnalysisSpecificSettings = None      # type: bool
    StatusMessages = None                   # type: ZOSAPI.Analysis.IMessages
    Title = None                            # type: str

    def Apply(self) -> 'ZOSAPI.Analysis.IMessage':
        pass

    def ApplyAndWaitForCompletion(self) -> 'ZOSAPI.Analysis.IMessage':
        pass

    def Close(self) -> None:
        pass

    def GetResults(self) -> 'ZOSAPI.Analysis.Data.IAR_':
        pass

    def GetSettings(self) -> 'ZOSAPI.Analysis.Settings.IAS_':
        pass

    def IsRunning(self) -> bool:
        pass

    def Release(self) -> None:
        pass

    def Terminate(self) -> bool:
        pass

    def ToFile(self, Filename: str, showSettings=False, verify=False) -> None:
        pass

    def WaitForCompletion(self) -> None:
        pass
