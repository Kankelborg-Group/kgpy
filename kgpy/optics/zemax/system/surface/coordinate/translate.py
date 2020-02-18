from astropy import units as u

from kgpy.optics.system import coordinate
from kgpy.optics.zemax.system.surface import Surface


class Translate(coordinate.Translate):

    _surface: Surface

    @property
    def z(self) -> u.Quantity:
        return self._z

    @z.setter
    def z(self, value: u.Quantity):
        self._z = value

        if value.isscalar:
            self._surface._lde_row.Thickness = value.to(self._surface._system.lens_units)