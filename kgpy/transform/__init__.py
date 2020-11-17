"""
Package for general coordinate transforms.
Currently, the only supported transforms are rigid transformations.
"""

__all__ = ['Transform', 'rigid']

from ._transform import Transform
from . import rigid

Transform.__module__ = __name__