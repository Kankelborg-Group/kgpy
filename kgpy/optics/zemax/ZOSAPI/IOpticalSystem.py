
from kgpy.optics.zemax import ZOSAPI

__all__ = ['IOpticalSystem']


class IOpticalSystem:

    Analyses = None             # type: ZOSAPI.Analysis.I_Analyses
    IsNonAxial = None           # type: bool
    LDE = None                  # type: ZOSAPI.Editors.LDE.ILensDataEditor
    MCE = None                  # type: ZOSAPI.Editors.MCE.I
    MFE = None                  # type: ZOSAPI.Editors.MFE.IMeritFunctionEditor
    Mode = None                 # type: ZOSAPI.SystemType
    NCE = None                  # type: ZOSAPI.Editors.NCE.INonSeqEditor
    NeedsSave = None            # type: bool
    SystemData = None           # type: ZOSAPI.SystemData.ISystemData
    SystemFile = None           # type: str
    SystemID = None             # type: int
    SystemName = None           # type: str
    TDE = None                  # type: ZOSAPI.Editors.TDE.IToleranceDataEditor
    TheApplication = None       # type: ZOSAPI.IZOSAPI_Application
    Tools = None                # type: ZOSAPI.Tools.IOpticalSystemTools

    def Close(self, saveIfNeeded: bool) -> bool:
        pass

    def CopySystem(self) -> 'IOpticalSystem':
        pass

    def GetCurrentStatus(self) -> str:
        pass

    def LoadFile(self, LensFile: str, saveIfNeeded: bool) -> bool:
        pass

    def MakeNonSequential(self) -> bool:
        pass

    def MakeSequential(self) -> bool:
        pass

    def New(self, saveIfNeeded: bool) -> None:
        pass

    def Save(self) -> None:
        pass

    def SaveAs(self, fileName: str) -> None:
        pass

    def UpdateStatus(self) -> str:
        pass
