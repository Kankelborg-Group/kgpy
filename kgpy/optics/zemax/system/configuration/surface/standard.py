
from kgpy import optics
from kgpy.optics.zemax import ZOSAPI

from . import Surface

__all__ = []


class Standard(optics.system.configuration.surface.Standard, Surface):

    def __init__(self, zemax_surface: ZOSAPI.Editors.LDE.ILDERow,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)





