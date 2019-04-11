
from typing import Union

__all__ = ['IMessages']


class IMessages:

    def AllToString(self) -> str:
        pass

    def WriteLine(self, s: str, userV: Union[bool, float, int, str], settingsV: Union[bool, float, int, str]) -> None:
        pass
