"""
Library of optical materials
"""

from ._material import Material
from ._mirror import Mirror

Material.__module__ = __name__
Mirror.__module__ = __name__
