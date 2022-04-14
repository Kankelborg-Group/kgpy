import dataclasses
import astropy.units as u
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vectors
import kgpy.optics.vectors


@dataclasses.dataclass(eq=False)
class ChannelComponents(kgpy.vectors.AbstractVector):
    spectral_order: kgpy.labeled.ArrayLike = 0
    angle_dispersion: kgpy.uncertainty.ArrayLike = 0


@dataclasses.dataclass(eq=False)
class PixelVector(
    kgpy.optics.vectors.PositionVector,
    ChannelComponents,
):
    pass

