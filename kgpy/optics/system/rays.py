import dataclasses
import typing as tp
import numpy as np
import astropy.units as u

__all__ = ['Rays']


class Axis:
    config = 0
    surf = 1
    wavl = 2
    field_x = ~3
    field_y = ~2
    pupil_x = ~1
    pupil_y = ~0


@dataclasses.dataclass
class Rays:
    axis = Axis()

    x: u.Quantity
    y: u.Quantity
    z: u.Quantity

    cos_ax: np.ndarray
    cos_ay: np.ndarray
    cos_az: np.ndarray

    cos_nx: np.ndarray
    cos_ny: np.ndarray
    cos_nz: np.ndarray

    opd: u.Quantity

    mask: np.ndarray

    @classmethod
    def empty(cls, sh: tp.Tuple[int, int, int, int, int, int, int]) -> 'Rays':
        return cls(
            x=np.empty(sh) * u.mm,
            y=np.empty(sh) * u.mm,
            z=np.empty(sh) * u.mm,
            cos_ax=np.empty(sh),
            cos_ay=np.empty(sh),
            cos_az=np.empty(sh),
            cos_nx=np.empty(sh),
            cos_ny=np.empty(sh),
            cos_nz=np.empty(sh),
            opd=np.empty(sh) * u.mm,
            mask=np.empty(sh, dtype=np.bool),
        )

    @property
    def pupil_average(self) -> 'Rays':
        ax = (self.axis.pupil_x, self.axis.pupil_y)

        return type(self)(
            x=np.mean(self.x.value, axis=ax, keepdims=True) * self.x.unit,
            y=np.mean(self.y.value, axis=ax, keepdims=True) * self.x.unit,
            z=np.mean(self.z.value, axis=ax, keepdims=True) * self.x.unit,
            cos_ax=np.mean(self.cos_ax, axis=ax, keepdims=True),
            cos_ay=np.mean(self.cos_ay, axis=ax, keepdims=True),
            cos_az=np.mean(self.cos_az, axis=ax, keepdims=True),
            cos_nx=np.mean(self.cos_nx, axis=ax, keepdims=True),
            cos_ny=np.mean(self.cos_ny, axis=ax, keepdims=True),
            cos_nz=np.mean(self.cos_nz, axis=ax, keepdims=True),
            opd=np.mean(self.opd.value, axis=ax, keepdims=True) * self.x.unit,
            mask=np.max(self.mask, axis=ax, keepdims=True)
        )
