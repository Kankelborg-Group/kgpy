"""
A package for finding the roots of scalar-valued functions.
"""

__all__ = ['secant', 'false_position']

from .secant import secant
from .false_position import false_position
