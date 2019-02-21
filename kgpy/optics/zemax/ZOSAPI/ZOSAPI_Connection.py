
from kgpy.optics.zemax import ZOSAPI

__all__ = ['ZOSAPI_Connection']


class ZOSAPI_Connection:

    IsAlive = None      # type: bool

    def ConnectToApplication(self) -> ZOSAPI.IZOSAPI_Application:
        pass

    def CreateNewApplication(self) -> ZOSAPI.IZOSAPI_Application:
        pass

    def CreateZemaxServer(self, applicationName: str) -> ZOSAPI.IZOSAPI_Application:
        pass