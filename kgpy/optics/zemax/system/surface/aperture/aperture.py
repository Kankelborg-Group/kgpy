import abc
import dataclasses
import typing as typ
import win32com.client
from kgpy.component import Component
from kgpy.optics.zemax import ZOSAPI
from .. import standard

__all__ = ['Aperture', 'NoAperture']


@dataclasses.dataclass
class Aperture(Component[standard.Standard]):

    @property
    @abc.abstractmethod
    def _lde_row_aperture_data(self) -> ZOSAPI.Editors.LDE.ISurfaceApertureType:
        return win32com.client.CastTo(
            self._composite._lde_row.ApertureData.CurrentTypeSettings,
            ZOSAPI.Editors.LDE.ISurfaceApertureType.__name__
        )


class NoAperture(Aperture):

    @property
    def _lde_row_aperture_data(self) -> ZOSAPI.Editors.LDE.ISurfaceApertureType:
        return super()._lde_row_aperture_data

    def _update(self) -> typ.NoReturn:
        super()._update()
