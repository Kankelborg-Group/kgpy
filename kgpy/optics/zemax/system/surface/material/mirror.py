import dataclasses
import typing as typ
from astropy import units as u
from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI
from . import material

__all__ = ['Mirror']


@dataclasses.dataclass
class Base:
    string: str = dataclasses.field(default='MIRROR', init=False, repr=False)


@dataclasses.dataclass
class Mirror(material.Mixin, system.surface.material.Mirror, Base):

    def _update(self) -> typ.NoReturn:
        super()._update()
        self.thickness = self.thickness

    @property
    def thickness(self) -> u.Quantity:
        return self._thickness

    @thickness.setter
    def thickness(self, value: u.Quantity):
        self._thickness = value
        try:
            self._composite.lde_row.DrawData.MirrorThickness = value.to(self._composite.lens_units).value
            self._composite.lde_row.DrawData.MirrorSubstrate = ZOSAPI.Editors.LDE.SubstrateType.Flat
        except AttributeError:
            pass
