import astropy.units as u

__all__ = ['quantity']


def quantity(
        a: u.Quantity,
        scientific_notation: bool = False,
        digits_after_decimal: int = 3
) -> str:
    estr = '{0.value:0.' + str(digits_after_decimal) + 'e} {0.unit:latex}'
    fstr = '{0.value:0.' + str(digits_after_decimal) + 'f} {0.unit:latex}'

    if not scientific_notation:
        return fstr.format(a)
    else:
        return estr.format(a)
