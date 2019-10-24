import dataclasses
import typing as tp
import nptyping as npt
import astropy.units as u

from . import Material, Aperture, Standard, CoordinateBreak

__all__ = ['MultiSurface']


@dataclasses.dataclass
class MultiSurface:

    coordinate_break_before: CoordinateBreak = CoordinateBreak()
    aperture_surface: Standard = Standard()
    main_surface: Standard = Standard()
    coordinate_break_after_z: CoordinateBreak = CoordinateBreak()
    coordinate_break_after: CoordinateBreak = CoordinateBreak()

    @classmethod
    def from_surface_params(
            cls,
            name: tp.Union[str, npt.Array[str]] = '',
            is_stop: tp.Union[bool, npt.Array[bool]] = False,
            thickness: u.Quantity = 0 * u.mm,
            radius: u.Quantity = 0 * u.mm,
            conic: u.Quantity = 0 * u.dimensionless_unscaled,
            material: tp.Union[tp.Optional[Material], npt.Array[tp.Optional[Material]]] = None,
            aperture: tp.Union[tp.Optional[Aperture], npt.Array[tp.Optional[Aperture]]] = None,
            mechanical_aperture: tp.Union[tp.Optional[Aperture], npt.Array[tp.Optional[Aperture]]] = None,
            decenter_before: u.Quantity = [0, 0, 0] * u.m,
            decenter_after: u.Quantity = [0, 0, 0] * u.m,
            tilt_before: u.Quantity = [0, 0, 0] * u.deg,
            tilt_after: u.Quantity = [0, 0, 0] * u.deg,
    ):

        cb_before = CoordinateBreak(
            name=name + '.cb_before',
            tilt=tilt_before,
            decenter=decenter_before,
        )

        aper_surf = Standard(
            name=name + '.aperture',
            aperture=aperture,
        )

        main_surf = Standard(
            name=name,
            is_stop=is_stop,
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
        decenter_after_z = decenter_after.copy()
        decenter_after_z[..., :~0] = 0
        decenter_after[..., ~0] = 0

        cb_after_z = CoordinateBreak(
            name=name + '.cb_after_z',
            decenter=decenter_after_z,
        )

        cb_after = CoordinateBreak(
            name=name+'.cb_after',
            thickness=thickness,
            tilt=tilt_after,
            decenter=decenter_after,
        )

        return cls(cb_before, aper_surf, main_surf, cb_after_z, cb_after)

    @property
    def name(self):
        return self.main_surface.name

    @property
    def is_stop(self):
        return self.main_surface.is_stop

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
