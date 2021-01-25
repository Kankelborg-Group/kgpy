import typing as typ
import dataclasses
import numpy as np
import astropy.units as u

__all__ = ['TolQuantity']


class TolQuantity(u.Quantity):

    def __new__(
            cls,
            *args,
            tmin: u.Quantity = 0 * u.dimensionless_unscaled,
            tmax: u.Quantity = 0 * u.dimensionless_unscaled,
            **kwargs
    ):
        self = super().__new__(cls, *args, **kwargs)
        self.tmin = tmin
        self.tmax = tmax
        return self

    @property
    def quantity(self) -> u.Quantity:
        return self.value << self.unit

    @property
    def amax(self):
        return self.quantity + self.tmax

    @property
    def amin(self):
        return self.quantity + self.tmin

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self.tmin = 0
        self.tmax = 0

    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        result = super().__array_ufunc__(function, method, *inputs, **kwargs)
        amin_inputs, amax_inputs = [], []
        for inp in inputs:
            if isinstance(inp, type(self)):
                amin_inputs.append(inp.amin)
                amax_inputs.append(inp.amax)
            else:
                amin_inputs.append(inp)
                amax_inputs.append(inp)

        if self.tmin != 0:
            amin = super().__array_ufunc__(function, method, *amin_inputs, **kwargs).quantity
            result.tmin = amin - result.quantity
        if self.tmax != 0:
            amax = super().__array_ufunc__(function, method, *amax_inputs, **kwargs).quantity
            result.tmax = amax - result.quantity
        return result

    def __quantity_subclass__(self, unit: u.UnitBase):
        return TolQuantity, True
