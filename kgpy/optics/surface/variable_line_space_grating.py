import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
from kgpy import vector, optics
from . import DiffractionGrating

__all__ = ['VariableLineSpaceGrating']

MaterialT = typ.TypeVar('MaterialT', bound=optics.Material)
ApertureT = typ.TypeVar('ApertureT', bound=optics.Aperture)
ApertureMechT = typ.TypeVar('ApertureMechT', bound=optics.Aperture)


@dataclasses.dataclass
class VariableLineSpaceGrating(DiffractionGrating[MaterialT, ApertureT, ApertureMechT]):

    coeff_linear: u.Quantity = 0 / (u.mm ** 2)
    coeff_quadratic: u.Quantity = 0 / (u.mm ** 3)
    coeff_cubic: u.Quantity = 0 / (u.mm ** 4)

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.coeff_linear,
            self.coeff_quadratic,
            self.coeff_cubic,
        )

    def groove_normal(self, sx: u.Quantity, sy: u.Quantity) -> u.Quantity:
        sx2 = np.square(sx)
        term0 = self.groove_density
        # term0 = 1 / term0
        term1 = self.coeff_linear * sx
        term2 = self.coeff_quadratic * sx2
        term3 = self.coeff_cubic * sx * sx2
        groove_density = term0 + term1 + term2 + term3
        # groove_density = 1 / terms
        return vector.from_components(x=groove_density)

    def copy(self) -> 'VariableLineSpaceGrating':
        other = super().copy()      # type: VariableLineSpaceGrating
        other.coeff_linear = self.coeff_linear.copy()
        other.coeff_quadratic = self.coeff_quadratic.copy()
        other.coeff_cubic = self.coeff_cubic.copy()
        return other
