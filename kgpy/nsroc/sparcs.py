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

    def view(self) -> 'SPARCS':
        other = super().view()      # type: SPARCS
        other.pointing_jitter = self.pointing_jitter
        other.pointing_drift = self.pointing_drift
        other.rlg_jitter = self.rlg_jitter
        other.rlg_drift = self.rlg_drift
        return other

    def copy(self) -> 'SPARCS':
        other = super().copy()      # type: SPARCS
        other.pointing_jitter = self.pointing_jitter.copy()
        other.pointing_drift = self.pointing_drift.copy()
        other.rlg_jitter = self.rlg_jitter.copy()
        other.rlg_drift = self.rlg_drift.copy()
        return other


def specification() -> 'SPARCS':
    sparcs = SPARCS()
    sparcs.pointing_jitter = 0.5 / np.sqrt(8) * u.arcsec
    sparcs.pointing_drift = 0.014 * u.arcsec / u.s
    sparcs.rlg_jitter = (0.01 * u.deg).to(u.arcsec)
    sparcs.rlg_drift = (0.04 * u.deg / u.hr).to(u.arcsec / u.s)
    return sparcs
