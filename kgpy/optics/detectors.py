from __future__ import annotations
from typing import TypeVar, Any
import dataclasses
import astropy.units as u
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vectors
import kgpy.function
from . import vectors
from . import materials
from . import apertures
from . import surfaces

MaterialT = TypeVar('MaterialT', bound=materials.Material)
DetectorT = TypeVar('DetectorT', bound='Detector')


@dataclasses.dataclass
class Detector(
    surfaces.Surface[None, MaterialT, apertures.Rectangular, apertures.Rectangular, None],
):

    shape_pixels: kgpy.vectors.Cartesian2D[int, int] = None

    @property
    def indices_pixels(self: DetectorT) -> kgpy.vectors.Cartesian2D:
        return kgpy.vectors.Cartesian2D(
            x=kgpy.labeled.Range(stop=self.shape_pixels.x, axis='detector_x'),
            y=kgpy.labeled.Range(stop=self.shape_pixels.y, axis='detector_y'),
        )

    @property
    def pitch_pixels(self: DetectorT) -> kgpy.vectors.Cartesian2D:
        return 2 * self.aperture.half_width / self.shape_pixels

    @property
    def position_pixels(self):
        return self.indices_pixels * self.pitch_pixels + self.pitch_pixels / 2

    def __call__(
            self,
            image: kgpy.function.Array[vectors.SpectralPositionVector, kgpy.uncertainty.ArrayLike],
            axis_field: str | list[str],
            axis_wavelength: str | list[str],
            wavelength_sum: bool = True
    ) -> kgpy.function.Array[vectors.SpectralPositionVector, kgpy.uncertainty.ArrayLike]:

        input_new = image.input.copy_shallow()
        for component in input_new.coordinates:
            input_new.coordinates[component] = None
        input_new.position_x = kgpy.labeled.LinearSpace(
            start=self.aperture.min.x,
            stop=self.aperture.max.x,
            num=self.shape_pixels.x,
            axis='detector_x',
        )
        input_new.position_y = kgpy.labeled.LinearSpace(
            start=self.aperture.min.y,
            stop=self.aperture.max.y,
            num=self.shape_pixels.y,
            axis='detector_y',
        )

        result = image.interp_linear(
            input_new=input_new,
            axis=axis_field,
        )

        if wavelength_sum:
            result.output = result.output.sum(axis_wavelength)

        return result


