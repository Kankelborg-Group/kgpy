import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
from .. import material, aperture
from . import DiffractionGrating

__all__ = ['VariableLineSpaceGrating']

MaterialT = typ.TypeVar('MaterialT', bound=material.Material)
ApertureT = typ.TypeVar('ApertureT', bound=aperture.Aperture)


@dataclasses.dataclass
class VariableLineSpaceGrating(DiffractionGrating[MaterialT, ApertureT]):

    coeff_linear: u.Quantity = 0
    coeff_quadratic: u.Quantity = 0
    coeff_cubic: u.Quantity = 0

    @property
    def __init__args(self) -> typ.Dict[str, typ.Any]:
        args = super().__init__args
        args.update({
            'coeff_linear': self.coeff_linear,
            'coeff_quadratic': self.coeff_quadratic,
            'coeff_cubic': self.coeff_cubic,
        })
        return args

    @property
    def config_broadcast(self):
        return np.broadcast(
            super().config_broadcast,
            self.diffraction_order,
            self.groove_frequency,
        )

    def groove_normal(self, sx: u.Quantity, sy: u.Quantity) -> u.Quantity:
        sx2 = np.square(sx)
        term0 = 1 / self.groove_frequency
        term1 = self.coeff_linear * sx
        term2 = self.coeff_quadratic * sx2
        term3 = self.coeff_cubic * sx * sx2
        groove_density = term0 + term1 + term2 + term3
        return u.Quantity([0 << 1 / u.mm, 1 / groove_density, 0 << 1 / u.mm])