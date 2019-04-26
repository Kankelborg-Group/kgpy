
from kgpy.optics.zemax import ZOSAPI

__all__ = []


class IBatchRayTrace:

    CanCancel = None                # type: bool
    ErrorMessage = None             # type: str
    IsAsynchronous = None           # type: bool
    IsFiniteDuration = None         # type: bool
    IsRunning = None                # type: bool
    IsValid = None                  # type: bool
    Progress = None                 # type: int
    Status = None                   # type: str
    Succeeded = None                # type: bool

    def Cancel(self) -> bool:
        pass

    def Close(self) -> bool:
        pass

    def CreateDirectPol(self, MaxRays: int, rayType: 'ZOSAPI.Tools.RayTrace.RaysType', Ex: float, Ey: float, phax: float,
                        phay: float, startSurface: int, toSurface: int) -> 'ZOSAPI.Tools.RayTrace.IRayTraceDirectPolData':
        pass

    def CreateDirectUnpol(self, MaxRays: int, rayType: 'ZOSAPI.Tools.RayTrace.RaysType', startSurface: int,
                          toSurface: int) -> 'ZOSAPI.Tools.RayTrace.IRayTraceDirectUnpolData':
        pass

    def CreateNormPol(self, MaxRays: int, rayType: 'ZOSAPI.Tools.RayTrace.RaysType', Ex: float, Ey: float, phax: float,
                        phay: float, startSurface: int, toSurface: int
                        ) -> 'ZOSAPI.Tools.RayTrace.IRayTraceNormPolData':
        pass

    def CreateNormUnpol(self, MaxRays: int, rayType: 'ZOSAPI.Tools.RayTrace.RaysType', toSurface: int
                        ) -> 'ZOSAPI.Tools.RayTrace.IRayTraceNormUnpolData':
        pass

    def CreateNSC(self):
        pass

    def CreateNSCSourceData(self):
        pass

    def GetDirectFieldCoordinates(self):
        pass

    def GetPhase(self):
        pass

    def Run(self):
        pass

    def RunAndWaitForCompletion(self) -> bool:
        pass

    def RunAndWaitWithTimeout(self):
        pass

    def SingleRayDirectPol(self):
        pass

    def SingleRayDirectPolFull(self):
        pass

    def SingleRayDirectUnpol(self):
        pass

    def SingleRayNormPol(self):
        pass

    def SingleRayNormPolFull(self):
        pass

    def SingleRayNormUnpol(self):
        pass

    def WaitForCompletion(self):
        pass

    def WaitWithTimeout(self):
        pass
