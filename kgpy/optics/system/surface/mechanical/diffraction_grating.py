import dataclasses
import typing as tp
import astropy.units as u

from kgpy.optics.system import surface
from . import Standard

__all__ = ['DiffractionGrating']


@dataclasses.dataclass
class DiffractionGrating(Standard):

    main_surface: tp.Union[surface.DiffractionGrating, 'DiffractionGrating'] = dataclasses.field(
        default_factory=lambda: surface.DiffractionGrating())

    @classmethod
    def from_surface_params(
            cls,
            diffraction_order: u.Quantity = 0 * u.dimensionless_unscaled,
            groove_frequency: u.Quantity = 0 * (1 / u.mm),
            **kwargs,
    ) -> 'DiffractionGrating':

        s = super().from_surface_params(**kwargs)

        main_surf = surface.DiffractionGrating(
            name=s.name,
            radius=s.radius,
            conic=s.conic,
            material=s.material,
            aperture=s.mechanical_aperture,
            diffraction_order=diffraction_order,
            groove_frequency=groove_frequency,
        )

        return cls(
            coordinate_break_before=s.coordinate_break_before,
            aperture_surface=s.aperture_surface,
            main_surface=main_surf,
            coordinate_break_after_z=s.coordinate_break_after_z,
            coordinate_break_after=s.coordinate_break_after
        )

    @property
    def diffraction_order(self):
        return self.main_surface.diffraction_order

    @property
    def groove_frequency(self):
        return self.main_surface.groove_frequency
