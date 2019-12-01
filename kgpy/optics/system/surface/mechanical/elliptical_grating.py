import dataclasses
import typing as tp
import astropy.units as u

from kgpy.optics.system import surface
from . import DiffractionGrating

__all__ = ['EllipticalGrating1']


@dataclasses.dataclass
class EllipticalGrating1(DiffractionGrating):

    main_surface: tp.Union[surface.EllipticalGrating1, 'EllipticalGrating1'] = dataclasses.field(
        default_factory=lambda: surface.EllipticalGrating1())

    @classmethod
    def from_surface_params(
            cls,
            a: u.Quantity = 0 * u.dimensionless_unscaled,
            b: u.Quantity = 0 * u.dimensionless_unscaled,
            c: u.Quantity = 0 * u.m,
            alpha: u.Quantity = 0 * u.dimensionless_unscaled,
            beta: u.Quantity = 0 * u.dimensionless_unscaled,
            **kwargs,
    ):

        s = super().from_surface_params(**kwargs)

        main_surf = surface.EllipticalGrating1(
            name=s.name,
            radius=s.radius,
            conic=s.conic,
            material=s.material,
            aperture=s.mechanical_aperture,
            diffraction_order=s.diffraction_order,
            groove_frequency=s.groove_frequency,
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
        )

        return cls(
            coordinate_break_before=s.coordinate_break_before,
            aperture_surface=s.aperture_surface,
            main_surface=main_surf,
            coordinate_break_after_z=s.coordinate_break_after_z,
            coordinate_break_after=s.coordinate_break_after
        )

    @property
    def a(self):
        return self.main_surface.a
    
    @property
    def b(self):
        return self.main_surface.b
    
    @property
    def c(self):
        return self.main_surface.c
    
    @property
    def alpha(self):
        return self.main_surface.alpha
    
    @property
    def beta(self):
        return self.main_surface.beta
