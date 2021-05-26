import ChiantiPy.core as ch
from dataclasses import dataclass
import kgpy.mixin


class Bunch(kgpy.mixin.Pickleable, ch.bunch):
    pass