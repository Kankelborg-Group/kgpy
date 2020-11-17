"""
kgpy root package
"""

__all__ = [
    'AutoAxis',
    'linspace', 'midspace',
    'Name',
    'fft',
    'rebin',
    'Obs',
]

from ._auto_axis import AutoAxis
from ._linspace import linspace, midspace
from ._name import Name
from . import fft
from ._rebin import rebin
from ._obs import Obs
