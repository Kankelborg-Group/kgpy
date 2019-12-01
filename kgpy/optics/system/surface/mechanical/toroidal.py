import dataclasses
import typing as tp
import astropy.units as u

from kgpy.optics.system import surface
from . import Standard

__all__ = ['Toroidal']


@dataclasses.dataclass
class Toroidal(Standard):

    main_surface: tp.Union[surface.Toroidal, 'Toroidal'] = dataclasses.field(default_factory=lambda: surface.Toroidal())

    @classmethod
    def from_surface_params(
            cls,
            radius_of_rotation=0 * u.mm,
            **kwargs,
    ):
        s = super().from_surface_params(**kwargs)

        main_surf = surface.Toroidal(
            name=s.name,
            is_stop=s.is_stop,
            is_detector=s.is_detector,
            radius=s.radius,
            conic=s.conic,
            material=s.material,
            aperture=s.mechanical_aperture,
            radius_of_rotation=radius_of_rotation
        )

        return cls(
            coordinate_break_before=s.coordinate_break_before,
            aperture_surface=s.aperture_surface,
            main_surface=main_surf,
            coordinate_break_after_z=s.coordinate_break_after_z,
            coordinate_break_after=s.coordinate_break_after
        )
