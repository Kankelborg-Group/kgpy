
from kgpy.optics.zemax import ZOSAPI

__all__ = ['IMessage']


class IMessage:

    ErrorCode = None    # type: ZOSAPI.Analysis.ErrorType
    Text = None         # type: str

