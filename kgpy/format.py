import typing as typ
import numpy as np
import astropy.units as u

__all__ = ['quantity']


def quantity(
        a: u.Quantity,
        scientific_notation: typ.Optional[bool] = None,
        digits_after_decimal: int = 3
) -> str:

    if a.ndim == 0:
        estr = '{0.value:0.' + str(digits_after_decimal) + 'e} {0.unit:latex_inline}'
        fstr = '{0.value:0.' + str(digits_after_decimal) + 'f} {0.unit:latex_inline}'

        if scientific_notation is None:
            if np.abs(a.value).any() > 0.1:
                scientific_notation = False
            else:
                scientific_notation = True

        if not scientific_notation:
            return fstr.format(a)
        else:
            return estr.format(a)

    else:
        return '{0} {1:latex_inline}'.format(np.array2string(
            a=a.value,
            precision=digits_after_decimal,
            separator=', ',
            floatmode='fixed'
        ), a.unit)
