"""
A package for finding the roots of scalar-valued functions.
"""

__all__ = ['secant', 'false_position']

from ._secant import secant
from ._false_position import false_position
