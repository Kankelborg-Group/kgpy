import typing as typ
import dataclasses
import numpy as np
import astropy.units as u

__all__ = ['TolQuantity']


class TolArray(np.ndarray):

    def __new__(
            cls,
            *args,
            vmin: np.ndarray = 0,
            vmax: np.ndarray = 0,
            **kwargs
    ):
        self = super().__new__(cls, *args, **kwargs)
        self.vmin = vmin
        self.vmax = vmax
        return self

    def __getitem__(self, item):
        other = super().__getitem__(item)
        vmin_item, vmax_item = item, item
        if isinstance(item, tuple):
            vmin_item, vmax_item = [], []
            for i in item:
                if isinstance(i, TolArray):
                    vmin_item.append(i.vmin)
                    vmax_item.append(i.vmax)
                else:
                    vmin_item.append(i)
                    vmax_item.append(i)
            vmin_item, vmax_item = tuple(vmin_item), tuple(vmax_item)
        elif isinstance(item, TolArray):
            vmin_item = item.vmin
            vmax_item = item.vmax
        other.vmin = self.vmin.__getitem__(vmin_item)
        other.vmax = self.vmax.__getitem__(vmax_item)
        return other

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        vmin_key, vmax_key = key, key
        vmin_value, vmax_value = value, value

        if isinstance(value, type(self)):
            vmin_value = value.vmin
            vmax_value = value.vmax
            if isinstance(key, tuple):
                vmin_key, vmax_key = [], []
                for k in key:
                    if isinstance(k, TolArray):
                        vmin_key.append(k.vmin)
                        vmax_key.append(k.vmax)
                    else:
                        vmin_key.append(k)
                        vmax_key.append(k)
                vmin_key, vmax_key = tuple(vmin_key), tuple(vmax_key)
                self.vmin.__setitem__(vmin_key, vmin_value)
                self.vmax.__setitem__(vmax_key, vmax_value)
            elif isinstance(key, TolArray):
                vmin_key = key.vmin
                vmax_key = key.vmax
                self.vmin.__setitem__(vmin_key, vmin_value)
                self.vmax.__setitem__(vmax_key, vmax_value)
        else:
            self.vmin.__setitem__(vmin_key, vmin_value)
            self.vmax.__setitem__(vmax_key, vmax_value)

    def __array_finalize__(self, obj):
        if super().__array_finalize__ is not None:
            super().__array_finalize__(obj)
        if obj is None:
            pass
        elif isinstance(obj, type(self)):
            self.vmin = obj.vmin.copy()
            self.vmax = obj.vmax.copy()
        else:
            self.vmin = self.view(self._default_type).copy()
            self.vmax = self.view(self._default_type).copy()

    @property
    def _default_type(self) -> typ.Type:
        return np.ndarray

    def _default_lim_factory(self):
        return np.array(self)

    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        # print(type(self), '__array_ufunc__', function)
        norm_inputs, vmin_inputs, vmax_outputs = [], [], []
        for inp in inputs:
            if isinstance(inp, type(self)):
                norm_inputs.append(inp.view(self._default_type))
                vmin_inputs.append(inp.vmin)
                vmax_outputs.append(inp.vmax)
            else:
                norm_inputs.append(inp)
                vmin_inputs.append(inp)
                vmax_outputs.append(inp)

        result = super().__array_ufunc__(function, method, *norm_inputs, **kwargs)
        if not isinstance(result, type(self)):
            result = np.asarray(result).view(TolArray)

        result.vmin = self.vmin.__array_ufunc__(function, method, *vmin_inputs, **kwargs)
        result.vmax = self.vmax.__array_ufunc__(function, method, *vmax_outputs, **kwargs)
        return result

    def __array_function__(self, function, types, args, kwargs):
        # print(type(self), '__array_function__', function)
        vmin_args, vmax_args = [], []
        for arg in args:
            if isinstance(arg, type(self)):
                vmin_arg = arg.vmin
                vmax_arg = arg.vmax
            elif isinstance(arg, list):
                vmin_arg, vmax_arg = [], []
                for a in arg:
                    if isinstance(a, type(self)):
                        vmin_arg.append(a.vmin)
                        vmax_arg.append(a.vmax)
                    else:
                        vmin_arg.append(a)
                        vmax_arg.append(a)
            else:
                vmin_arg = arg
                vmax_arg = arg
            vmin_args.append(vmin_arg)
            vmax_args.append(vmax_arg)
        result = super().__array_function__(function, types, args, kwargs)

        if isinstance(result, type(self)):
            result.vmin = self.vmin.__array_function__(function, types, tuple(vmin_args), kwargs)
            result.vmax = self.vmax.__array_function__(function, types, tuple(vmax_args), kwargs)

        elif isinstance(result, list):
            vmin = self.vmin.__array_function__(function, types, tuple(vmin_args), kwargs)
            vmax = self.vmax.__array_function__(function, types, tuple(vmax_args), kwargs)
            for result_i, vmin_i, vmax_i in zip(result, vmin, vmax):
                if isinstance(result_i, type(self)):
                    result_i.vmin = vmin_i
                    result_i.vmax = vmax_i

        return result

    def __repr__(self):
        return super().__repr__() + self.vmin.__repr__() + self.vmax.__repr__()


class TolQuantity(
    TolArray,
    u.Quantity,
):

    # __array_priority__ = 100000

    def __new__(
            cls,
            *args,
            vmin: u.Quantity = 0 * u.dimensionless_unscaled,
            vmax: u.Quantity = 0 * u.dimensionless_unscaled,
            **kwargs
    ):
        return super().__new__(cls, *args, vmin=vmin, vmax=vmax, **kwargs)
        # self = super().__new__(cls, *args, **kwargs)
        # self.vmin = vmin
        # self.vmax = vmax
        # return self

    @property
    def quantity(self) -> u.Quantity:
        return self.value << self.unit

    @property
    def _default_type(self) -> typ.Type:
        return u.Quantity

    def _default_lim_factory(self):
        return u.Quantity(self)

    # def __getitem__(self, item):
    #     other = super().__getitem__(item)
    #     if isinstance(item, TolArray):
    #         other.vmin = self.vmin.__getitem__(item.vmin)
    #         other.vmax = self.vmax.__getitem__(item.vmax)
    #     else:
    #         other.vmin = self.vmin.__getitem__(item)
    #         other.vmax = self.vmax.__getitem__(item)
    #     return other
    #
    # def __setitem__(self, key, value):
    #     super().__setitem__(key, value)
    #     vmin_key, vmax_key = key, key
    #     vmin_value, vmax_value = value, value
    #     if isinstance(key, TolArray):
    #         vmin_key = key.vmin
    #         vmax_key = key.vmax
    #     if isinstance(value, type(self)):
    #         vmin_value = value.vmin
    #         vmax_value = value.vmax
    #     self.vmin.__setitem__(vmin_key, vmin_value)
    #     self.vmax.__setitem__(vmax_key, vmax_value)
    #
    # def __array_finalize__(self, obj):
    #     super().__array_finalize__(obj)
    #     # print('__array_finalize__')
    #     # print(self, obj)
    #     if obj is None:
    #         pass
    #
    #     elif isinstance(obj, type(self)):
    #         # pass
    #         self.vmin = obj.vmin
    #         self.vmax = obj.vmax
    #     else:
    #         self.vmin = u.Quantity(value=self.value, unit=self.unit)
    #         self.vmax = u.Quantity(value=self.value, unit=self.unit)
    #
    # def __array_ufunc__(self, function, method, *inputs, **kwargs):
    #     # print('__array_ufunc__', function)
    #     mid_inputs, amin_inputs, amax_inputs = [], [], []
    #     for inp in inputs:
    #         if isinstance(inp, type(self)):
    #             mid_inputs.append(inp.quantity)
    #             amin_inputs.append(inp.vmin)
    #             amax_inputs.append(inp.vmax)
    #         else:
    #             mid_inputs.append(inp)
    #             amin_inputs.append(inp)
    #             amax_inputs.append(inp)
    #
    #     result = super().__array_ufunc__(function, method, *inputs, **kwargs)
    #     if not isinstance(result, type(self)):
    #         result = TolArray(result)
    #
    #     result.vmin = self.vmin.__array_ufunc__(function, method, *amin_inputs, **kwargs)
    #     result.vmax = self.vmax.__array_ufunc__(function, method, *amax_inputs, **kwargs)
    #     return result
    #
    # def __array_function__(self, function, types, args, kwargs):
    #     # print('__array_function__', function)
    #     vmin_args, vmax_args = [], []
    #     for arg in args:
    #         if isinstance(arg, type(self)):
    #             vmin_arg = arg.vmin
    #             vmax_arg = arg.vmax
    #         elif isinstance(arg, list):
    #             vmin_arg, vmax_arg = [], []
    #             for a in arg:
    #                 if isinstance(a, type(self)):
    #                     vmin_arg.append(a.vmin)
    #                     vmax_arg.append(a.vmax)
    #                 else:
    #                     vmin_arg.append(a)
    #                     vmax_arg.append(a)
    #         else:
    #             vmin_arg = arg
    #             vmax_arg = arg
    #         vmin_args.append(vmin_arg)
    #         vmax_args.append(vmax_arg)
    #     result = super().__array_function__(function, types, args, kwargs)
    #     if not isinstance(result, type(self)):
    #         return result
    #
    #     result.vmin = self.vmin.__array_function__(function, types, tuple(vmin_args), kwargs)
    #     result.vmax = self.vmax.__array_function__(function, types, tuple(vmax_args), kwargs)
    #     return result
    #
    def reshape(self, shape, order='C'):
        result = super().reshape(shape, order=order)
        result.vmin = self.vmin.reshape(shape, order=order)
        result.vmax = self.vmax.reshape(shape, order=order)
        return result

    @property
    def T(self, *axes):
        result = super().transpose(*axes)
        result.vmin = self.vmin.transpose(*axes)
        result.vmax = self.vmax.transpose(*axes)
        return result


    # def __array_wrap__(self, obj, context=None):
    #     print('__array_wrap__')
    #
    def __quantity_subclass__(self, unit: u.UnitBase):
        # print('__quantity_subclass__')
        return TolQuantity, True
