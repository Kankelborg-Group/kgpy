"""
Package for general coordinate transforms.
Currently, the only supported transforms are rigid transformations.
"""

__all__ = ['Transform', 'rigid']

from .transform import Transform
from . import rigid
