import dataclasses
from ..... import coordinate
from ... import standard
from . import tilt as tilt_, decenter as decenter_, tilt_first as tilt_first_

__all__ = ['TiltDecenter']


@dataclasses.dataclass
class Base:

    tilt: tilt_.Tilt = dataclasses.field(default_factory=lambda: tilt_.Tilt())
    decenter: decenter_.Decenter = dataclasses.field(default_factory=lambda: decenter_.Decenter())
    tilt_first: tilt_first_.TiltFirst = dataclasses.field(default_factory=lambda: tilt_first_.TiltFirst())


# noinspection PyDataclass
@dataclasses.dataclass
class TiltDecenter(coordinate.TiltDecenter[standard.Standard], Base):
    pass
