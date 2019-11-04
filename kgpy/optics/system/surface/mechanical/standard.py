import dataclasses
import typing as tp
import nptyping as npt
import numpy as np
import astropy.units as u

from kgpy.optics.system import surface

__all__ = ['Standard']


@dataclasses.dataclass
class Standard:

    coordinate_break_before: surface.CoordinateBreak = surface.CoordinateBreak()
    aperture_surface: surface.Standard = surface.Standard()
    main_surface: tp.Union[surface.Standard, 'Standard'] = surface.Standard()
    coordinate_break_after_z: surface.CoordinateBreak = surface.CoordinateBreak()
    coordinate_break_after: surface.CoordinateBreak = surface.CoordinateBreak()

    @classmethod
    def from_surface_params(
            cls,
            name: tp.Union[str, npt.Array[str]] = '',
            is_stop: tp.Union[bool, npt.Array[bool]] = False,
            is_detector: tp.Union[bool, npt.Array[bool]] = False,
            thickness: u.Quantity = 0 * u.mm,
            radius: u.Quantity = np.inf * u.mm,
            conic: u.Quantity = 0 * u.dimensionless_unscaled,
            material: tp.Union[tp.Optional[surface.Material], npt.Array[tp.Optional[surface.Material]]] = None,
            aperture: tp.Union[tp.Optional[surface.Aperture], npt.Array[tp.Optional[surface.Aperture]]] = None,
            mechanical_aperture: tp.Union[tp.Optional[surface.Aperture],
                                          npt.Array[tp.Optional[surface.Aperture]]] = None,
            decenter_before: u.Quantity = [0, 0, 0] * u.m,
            decenter_after: u.Quantity = [0, 0, 0] * u.m,
            tilt_before: u.Quantity = [0, 0, 0] * u.deg,
            tilt_after: u.Quantity = [0, 0, 0] * u.deg,
            tilt_first_before: bool = False,
            tilt_first_after: bool = True,
    ) -> 'Standard':

        cb_before = surface.CoordinateBreak(
            name=name + '.cb_before',
            tilt=tilt_before,
            decenter=decenter_before,
            thickness=decenter_before[..., ~0],
            tilt_first=tilt_first_before
        )

        aper_surf = surface.Standard(
            name=name + '.aperture',
            aperture=aperture,
        )

        main_surf = surface.Standard(
            name=name,
            radius=radius,
            conic=conic,
            material=material,
            aperture=mechanical_aperture,
        )

        # Zemax doesn't allow decenters in z, instead you're supposed to use the `thickness` parameter.
        # Unfortunately, the `thickness` parameter is always applied last, regardless of the `order` parameter of the
        # surface.
        # As a workaround, we can add an extra surface to take care of the translation in z, before the rest of the
        # tilts/decenters are applied.
        # decenter_after_z = decenter_after.copy()
        # decenter_after_z[..., :~0] = 0
        # decenter_after[..., ~0] = 0

        cb_after_z = surface.CoordinateBreak(
            name=name + '.cb_after_z',
            thickness=decenter_after[..., ~0],
        )

        cb_after = surface.CoordinateBreak(
            name=name+'.cb_after',
            thickness=thickness,
            tilt=tilt_after,
            decenter=decenter_after,
            tilt_first=tilt_first_after,
        )

        return cls(cb_before, aper_surf, main_surf, cb_after_z, cb_after)
    
    @property
    def surfaces(self) -> tp.List[surface.Surface]:
        return [
            self.coordinate_break_before,
            self.aperture_surface,
            self.main_surface,
            self.coordinate_break_after_z,
            self.coordinate_break_after,
        ]

    @property
    def name(self):
        return self.main_surface.name

    @property
    def thickness(self):
        return self.coordinate_break_after.thickness

    @property
    def radius(self):
        return self.main_surface.radius

    @property
    def conic(self):
        return self.main_surface.conic

    @property
    def material(self):
        return self.main_surface.material

    @property
    def aperture(self):
        return self.aperture_surface.aperture

    @property
    def mechanical_aperture(self):
        return self.main_surface.aperture

    @property
    def decenter_before(self):
        return self.coordinate_break_before.decenter

    @property
    def decenter_after(self):
        return self.coordinate_break_after.decenter + self.coordinate_break_after_z.decenter

    @property
    def tilt_before(self):
        return self.coordinate_break_before.tilt

    @property
    def tilt_after(self):
        return self.coordinate_break_after.tilt
