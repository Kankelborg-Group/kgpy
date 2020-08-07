import typing as typ
import dataclasses
from .. import material, aperture
from . import Toroidal, VariableLineSpaceGrating

__all__ = ['ToroidalVariableLineSpaceGrating']

MaterialT = typ.TypeVar('MaterialT', bound=material.Material)
ApertureT = typ.TypeVar('ApertureT', bound=aperture.Aperture)


@dataclasses.dataclass
class ToroidalVariableLineSpaceGrating(VariableLineSpaceGrating[MaterialT, ApertureT], Toroidal[MaterialT, ApertureT]):
    pass
