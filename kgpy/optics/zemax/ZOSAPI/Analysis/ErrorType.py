
from win32com.client import constants

__all__ = ['ErrorType']


class ErrorTypeBase:

    @property
    def Success(self) -> 'ErrorType':
        return constants.ErrorType_Success

    @property
    def InvalidParameter(self) -> 'ErrorType':
        return constants.ErrorType_InvalidParameter

    @property
    def InvalidSettings(self) -> 'ErrorType':
        return constants.ErrorType_InvalidSettings

    @property
    def Failed(self) -> 'ErrorType':
        return constants.ErrorType_Failed

    @property
    def AnalysisUnavailableForProgramMode(self) -> 'ErrorType':
        return constants.ErrorType_AnalysisUnavailableForProgramMode

    @property
    def NotYetImplemented(self) -> 'ErrorType':
        return constants.ErrorType_NotYetImplemented


ErrorType = ErrorTypeBase()