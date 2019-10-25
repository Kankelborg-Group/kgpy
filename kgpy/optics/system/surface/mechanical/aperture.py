import dataclasses
import typing as tp
import nptyping as npt
import astropy.units as u

from kgpy.optics.system import surface

__all__ = ['Aperture', 'Circular', 'Rectangular', 'RegularPolygon', 'Spider']


@dataclasses.dataclass
class Aperture:

    front_surface: surface.Standard
    back_surface: surface.Standard

    @classmethod
    def from_surface_params(
            cls,
            name: tp.Union[str, npt.Array[str]] = '',
            is_stop: tp.Union[bool, npt.Array[bool]] = False,
            thickness: u.Quantity = 0 * u.mm,
            aperture: tp.Union[tp.Optional[surface.Aperture], npt.Array[tp.Optional[surface.Aperture]]] = None
    ):

        front = surface.Standard(
            name=name+'.front',
            is_stop=is_stop,
            thickness=thickness,
            aperture=aperture
        )

        back = surface.Standard(
            name=name+'.back',
            aperture=aperture
        )

        return cls(front, back)

    @classmethod
    def from_aper_params(
            cls,
            decenter_x: u.Quantity = 0 * u.m,
            decenter_y: u.Quantity = 0 * u.m,
            is_obscuration: tp.Union[bool, npt.Array[bool]] = False,
            **kwargs,
    ):

        aper = surface.Aperture(decenter_x=decenter_x, decenter_y=decenter_y, is_obscuration=is_obscuration)

        return cls.from_surface_params(aperture=aper, **kwargs)

    @property
    def decenter_x(self):
        return self.front_surface.aperture.decenter_x

    @property
    def decenter_y(self):
        return self.front_surface.aperture.decenter_y

    @property
    def is_obscuration(self):
        return self.front_surface.aperture.is_obscuration


class Circular(Aperture):

    @classmethod
    def from_aper_params(
            cls,
            decenter_x: u.Quantity = 0 * u.m,
            decenter_y: u.Quantity = 0 * u.m,
            is_obscuration: tp.Union[bool, npt.Array[bool]] = False,
            radius: u.Quantity = 0 * u.m,
            **kwargs,
    ):

        aper = surface.aperture.Circular(decenter_x=decenter_x, decenter_y=decenter_y, is_obscuration=is_obscuration,
                                         radius=radius)

        return cls.from_surface_params(aperture=aper, **kwargs)

    @property
    def radius(self):
        return self.front_surface.aperture.radius


class Rectangular(Aperture):

    @classmethod
    def from_aper_params(
            cls,
            decenter_x: u.Quantity = 0 * u.m,
            decenter_y: u.Quantity = 0 * u.m,
            is_obscuration: tp.Union[bool, npt.Array[bool]] = False,
            radius: u.Quantity = 0 * u.m,
            **kwargs,
    ):

        aper = surface.aperture.Circular(decenter_x=decenter_x, decenter_y=decenter_y, is_obscuration=is_obscuration,
                                         radius=radius)

        return cls.from_surface_params(aperture=aper, **kwargs)



class RegularPolygon(Aperture):
    pass


class Spider(Aperture):
    pass
