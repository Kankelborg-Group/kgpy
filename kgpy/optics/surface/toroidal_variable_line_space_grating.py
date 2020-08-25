import typing as typ
import dataclasses
from kgpy import optics
from . import Toroidal, VariableLineSpaceGrating

__all__ = ['ToroidalVariableLineSpaceGrating']

MaterialT = typ.TypeVar('MaterialT', bound=optics.Material)
ApertureT = typ.TypeVar('ApertureT', bound=optics.Aperture)
ApertureMechT = typ.TypeVar('ApertureMechT', bound=optics.Aperture)


@dataclasses.dataclass
class ToroidalVariableLineSpaceGrating(
    VariableLineSpaceGrating[MaterialT, ApertureT, ApertureMechT],
    Toroidal[MaterialT, ApertureT, ApertureMechT]
):
    pass
