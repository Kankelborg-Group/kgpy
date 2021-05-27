import ChiantiPy.core as ch
import kgpy.mixin
from collections import OrderedDict


class Bunch(kgpy.mixin.Pickleable, ch.bunch):
    pass


def write_roman(num):
    roman = OrderedDict()
    roman[40] = "xl"
    roman[10] = "x"
    roman[9] = "ix"
    roman[5] = "v"
    roman[4] = "iv"
    roman[1] = "i"

    def roman_num(num):
        for r in roman.keys():
            x, y = divmod(num, r)
            yield roman[r] * x
            num -= (r * x)
            if num <= 0:
                break

    return "".join([a for a in roman_num(num)])


def ion_tolatex(ions: str, use_latex:bool = True) -> str:
    ion_latex = []
    for ion in ions:
        element, ion = ion.split('_')
        if use_latex:
            ion_latex.append(element[0].upper() + element[1:] + '\,{\sc ' + write_roman(int(ion)) + '}')
        else:
            ion_latex.append(element[0].upper() + element[1:] + ' ' + write_roman(int(ion)).upper())

    return ion_latex
