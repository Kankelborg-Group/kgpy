import dataclasses
import typing as tp
import numpy as np
import astropy.units as u

__all__ = ['Rays']


class Axis:
    config = 0
    surf = 1
    wavl = 2
    field_x = ~4
    field_y = ~3
    pupil_x = ~2
    pupil_y = ~1


@dataclasses.dataclass
class Rays:
    axis = Axis()

    position: u.Quantity
    direction: np.ndarray
    surface_normal: np.ndarray

    opd: u.Quantity
    mask: np.ndarray

    @classmethod
    def empty(cls, sh: tp.Tuple[int, int, int, int, int, int, int]) -> 'Rays':

        vsh = sh + (3,)
        ssh = sh + (1,)

        return cls(
            position=np.empty(vsh) * u.mm,
            direction=np.empty(vsh),
            surface_normal=np.empty(vsh),
            opd=np.empty(ssh) * u.mm,
            mask=np.empty(ssh, dtype=np.bool),
        )

    @property
    def pupil_average(self) -> 'Rays':
        ax = (self.axis.pupil_x, self.axis.pupil_y)

        return type(self)(
            position=np.mean(self.position.value, axis=ax, keepdims=True) * self.position.unit,
            direction=np.mean(self.direction.value, axis=ax, keepdims=True) * self.direction.unit,
            surface_normal=np.mean(self.surface_normal.value, axis=ax, keepdims=True) * self.surface_normal.unit,
            opd=np.mean(self.opd.value, axis=ax, keepdims=True) * self.opd.unit,
            mask=np.max(self.mask, axis=ax, keepdims=True)
        )
