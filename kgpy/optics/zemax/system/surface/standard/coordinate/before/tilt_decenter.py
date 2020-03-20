import dataclasses
from ... import standard
from .... import coordinate
from . import Tilt, Decenter, TiltFirst

__all__ = ['TiltDecenter']


@dataclasses.dataclass
class Base:

    tilt: Tilt = dataclasses.field(default_factory=lambda: Tilt())
    decenter: Decenter = dataclasses.field(default_factory=lambda: Decenter())
    tilt_first: TiltFirst = dataclasses.field(default_factory=lambda: TiltFirst())


@dataclasses.dataclass
class TiltDecenter(coordinate.TiltDecenter[standard.Standard], Base):
    pass
