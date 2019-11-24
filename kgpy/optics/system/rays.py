import dataclasses
import typing as tp
import numpy as np
import astropy.units as u
import vg

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

    def unvignetted_mean(self, axis=None) -> 'Rays':

        uvm = self.unvignetted_mask

        norm = np.sum(uvm, axis=axis, keepdims=True)

        return type(self)(
            position=np.nansum(self.position.value * uvm, axis=axis, keepdims=True) * self.position.unit / norm,
            direction=np.nansum(self.direction * uvm, axis=axis, keepdims=True) / norm,
            surface_normal=np.nansum(self.surface_normal * uvm, axis=axis, keepdims=True) / norm,
            opd=np.nansum(self.opd.value * uvm, axis=axis, keepdims=True) * self.opd.unit / norm,
            mask=np.max(self.mask, axis=axis, keepdims=True)
        )

    @property
    def unvignetted_mask(self):
        sl = [slice(None)] * len(self.mask.shape)
        sl[self.axis.surf] = slice(~0, None)
        unvignetted_mask = self.mask[sl]
        return unvignetted_mask

    @property
    def pupil_mean(self):
        return self.unvignetted_mean(axis=(self.axis.pupil_x, self.axis.pupil_y))
    
    @property
    def field_mean(self):
        return self.unvignetted_mean(axis=(self.axis.field_x, self.axis.field_y))
    
    @property
    def incidence_angle(self):
        return self._angle_between_ray_and_surface(-1)
    
    @property
    def reflected_angle(self):
        return self._angle_between_ray_and_surface(0)
    
    def _angle_between_ray_and_surface(self, offset: int):
        
        sl = [slice(None)] * len(self.direction.shape)
        sl[self.axis.surf] = slice(1, None)

        normal = np.roll(self.surface_normal, offset, axis=self.axis.surf)
        direction = self.direction

        sh = normal.shape

        normal = -np.reshape(normal, (-1, 3))
        direction = np.reshape(direction, (-1, 3))

        x = vg.angle(normal, direction) * u.deg

        x = np.reshape(x, sh[:~0])
        x = x.squeeze()

        return x

