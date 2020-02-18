import dataclasses
import typing as typ

from ... import ZOSAPI
from ..system import System
from . import Surface

__all__ = ['Editor']


@dataclasses.dataclass
class Base:

    _surfaces: typ.List
    system: System

class Editor(Base):
    pass
