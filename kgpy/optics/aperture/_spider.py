import dataclasses
import abc
import typing as typ
import numpy as np
import astropy.units as u

from . import Aperture, Decenterable

__all__ = ['Spider']


@dataclasses.dataclass
class Spider(Decenterable, Aperture):

    arm_half_width: u.Quantity = 0 * u.mm
    num_arms: int = 2
    radius: u.Quantity = 0 * u.mm

    @property
    def broadcasted(self):
        out = super().broadcasted
        out = np.broadcast(out, self.arm_half_width)
        out = np.broadcast(out, self.radius)
        return out

    @property
    def wire(self) -> u.Quantity:
        a = np.linspace(0 * u.deg, 360 * u.deg, self.num_arms, endpoint=False)
        a = np.expand_dims(a, ~0)

        x = u.Quantity([0 * u.m, self.radius, self.radius, 0 * u.m])
        y = u.Quantity([-self.arm_half_width, -self.arm_half_width, self.arm_half_width, self.arm_half_width])

        xp = x * np.cos(a) - y * np.sin(a)
        yp = x * np.sin(a) + y * np.cos(a)

        pts = u.Quantity([xp, yp])
        pts = np.moveaxis(pts, 0, ~0)
        return pts