import typing as typ
import numpy as np
import astropy.units as u

__all__ = ['quantity']


def quantity(
        a: u.Quantity,
        scientific_notation: typ.Optional[bool] = None,
        digits_after_decimal: int = 3
) -> str:
    estr = '{0.value:0.' + str(digits_after_decimal) + 'e}$\,${0.unit:latex_inline}'
    fstr = '{0.value:0.' + str(digits_after_decimal) + 'f}$\,${0.unit:latex_inline}'

    if scientific_notation is None:
        if np.abs(a.value).any() > 0.1:
            scientific_notation = False
        else:
            scientific_notation = True

    if a.ndim == 0:

        if not scientific_notation:
            return fstr.format(a)
        else:
            if a != 0:
                unit = a.unit
                exponent = np.floor(np.log10(np.abs(a.value)))
                mantissa = a / 10 ** exponent
                format_str = '${0:0.' + str(digits_after_decimal) + 'f} \\times 10^{{{1}}}\\,${2:latex_inline}'
                return format_str.format(mantissa.value, exponent.astype(np.int), unit)
            else:
                return fstr.format(a)

    else:

        base_str = '{:0.' + str(digits_after_decimal)
        if not scientific_notation:
            formatter = base_str + 'f}'
        else:
            formatter = base_str + 'e}'

        return '{0} {1:latex_inline}'.format(np.array2string(
            a=a.value,
            precision=digits_after_decimal,
            separator=', ',
            floatmode='fixed',
            max_line_width=200,
            formatter=dict(float_kind=formatter.format)
        ), a.unit)
