import dataclasses
from kgpy.optics.system import coordinate
from ..surface import Surface

__all__ = ['Transform']


@dataclasses.dataclass
class Base(coordinate.Transform):

    surface: Surface = None


class Transform(Base):
    pass
