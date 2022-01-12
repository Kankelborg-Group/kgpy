import dataclasses
import astropy.units as u
import numpy as np

import kgpy.mixin

__all__ = ['SPARCS']


@dataclasses.dataclass
class SPARCS(kgpy.mixin.Copyable):
    pointing_jitter: u.Quantity = 0 * u.arcsec
    pointing_drift: u.Quantity = 0 * u.arcsec / u.s
    rlg_jitter: u.Quantity = 0 * u.arcsec
    rlg_drift: u.Quantity = 0 * u.arcsec / u.s


def specification() -> 'SPARCS':
    sparcs = SPARCS()
    sparcs.pointing_jitter = 0.5 * u.arcsec
    sparcs.pointing_drift = 0.014 * u.arcsec / u.s
    sparcs.rlg_jitter = (0.01 * u.deg).to(u.arcsec)
    sparcs.rlg_drift = (0.04 * u.deg / u.hr).to(u.arcsec / u.s)
    return sparcs
