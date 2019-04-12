
from typing import List

from kgpy.optics.zemax import ZOSAPI

__all__ = []


class IOpticalSystemTools:

    CurrentTool = None      # type: ZOSAPI.Tools.ISystemTool
    IsRunning = None        # type: bool
    Progress = None         # type: int
    Status = None           # type: str

    def CreateDoubleMatrix(self, Rows: int, Cols: int) -> List[List[float]]:
        pass

    def CreateDoubleVector(self, numElements: int) -> List[float]:
        pass

    def GetConversionFromSystemUnits(self, toUnits: ZOSAPI.SystemData.ZemaxSystemUnits) -> float:
        pass

    def GetConversionToSystemUnits(self, fromUnits: ZOSAPI.SystemData.ZemaxSystemUnits):
        pass

    def OpenBatchRayTrace(self):
        pass

    def OpenConvertToNSCGroup(self):
        pass

    def OpenCreateZAR(self):
        pass

    def OpenCriticalRaysetGenerator(self):
        pass

    def OpenDesignLockdown(self):
        pass

    def OpenExportCAD(self):
        pass

    def OpenGlobalOptimization(self):
        pass

    def OpenHammerOptimization(self):
        pass

    def OpenLensCatalogs(self):
        pass

    def OpenLightningTrace(self):
        pass

    def OpenLocalOptimization(self):
        pass

    def OpenMeritFunctionCalculator(self):
        pass

    def OpenNSCRatTrace(self):
        pass

    def OpenQuickAdjust(self):
        pass

    def OpenQuickFocus(self):
        pass

    def OpenRayDatabaseReader(self):
        pass

    def OpenRestoreZAR(self):
        pass

    def OpenRMSSpotRadiusCalculator(self):
        pass

    def OpenScale(self):
        pass

    def OpenTolerancing(self):
        pass

    def RemoveAllVariables(self):
        pass

    def SetAllRadiiVariable(self):
        pass

    def SetAllThicknessesVariable(self):
        pass