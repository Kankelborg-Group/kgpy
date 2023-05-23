from __future__ import annotations
import typing as typ
import abc
import dataclasses
import kgpy.labeled
import kgpy.uncertainty
import kgpy.function
import kgpy.optics
import kgpy.solar
from . import overlappograms

__all__ = [
    'AbstractInstrument',
    'AbstractSystemInstrument',
    'SystemInstrument',
]

AbstractInstrumentT = typ.TypeVar('AbstractInstrumentT', bound='AbstractInstrument')
AbstractSystemInstrumentT = typ.TypeVar('AbstractSystemInstrumentT', bound='AbstractSystemInstrument')
SystemInstrumentT = typ.TypeVar('SystemInstrumentT', bound='SystemInstrument')


@dataclasses.dataclass
class AbstractInstrument(
    abc.ABC,
):
    """
    Interface describing the forward model of a CT imaging spectrograph.
    """

    @abc.abstractmethod
    def __call__(
            self: AbstractInstrumentT,
            scene: kgpy.solar.SpectralRadiance,
            axis_field: str | list[str],
            axis_wavelength: str | list[str],
            wavelength_sum: bool = True,
    ) -> overlappograms.Overlappogram:
        pass

    @abc.abstractmethod
    def deproject(
            self: AbstractInstrumentT,
            image: overlappograms.Overlappogram,
    ) -> kgpy.solar.SpectralRadiance:
        pass


@dataclasses.dataclass
class AbstractSystemInstrument(
    AbstractInstrument,
):

    @property
    @abc.abstractmethod
    def system(self: AbstractSystemInstrumentT) -> kgpy.optics.systems.AbstractSystem:
        pass

    def __call__(
            self: AbstractSystemInstrumentT,
            scene: kgpy.solar.SpectralRadiance,
            axis_field: str | list[str],
            axis_wavelength: str | list[str],
            wavelength_sum: bool = True,
    ) -> overlappograms.Overlappogram:
        return self.system(
            scene=scene,
            axis_field=axis_field,
            axis_wavelength=axis_wavelength,
            wavelength_sum=wavelength_sum,
        )

    def deproject(
            self: AbstractSystemInstrumentT,
            image: overlappograms.Overlappogram,
    ) -> kgpy.solar.SpectralRadiance:
        return self.system.inverse(image)


@dataclasses.dataclass
class SystemInstrument(
    AbstractSystemInstrument
):

    system: kgpy.optics.systems.AbstractSystem = None


