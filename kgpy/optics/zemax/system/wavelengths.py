import typing as tp
import numpy as np
import astropy.units as u

from kgpy import optics
from kgpy.optics.zemax import ZOSAPI

from . import util

zemax_wavelength_units = u.um


def add_to_zemax_system(
        zemax_system: ZOSAPI.IOpticalSystem,
        wavelengths: 'optics.system.Wavelengths',
        configuration_shape: tp.Tuple[int],
):
    
    while zemax_system.SystemData.Wavelengths.NumberOfWavelengths < wavelengths.num_per_config:
        zemax_system.SystemData.Wavelengths.AddWavelength(1, 1)
    
    csh = configuration_shape
    
    op_wave = ZOSAPI.Editors.MCE.MultiConfigOperandType.WAVE
    op_weight = ZOSAPI.Editors.MCE.MultiConfigOperandType.WLWT

    unit_wave = u.um
    unit_weight = u.dimensionless_unscaled

    sh = (wavelengths.num_per_config,)

    wavls = np.broadcast_to(wavelengths.values, sh) * wavelengths.values.unit
    weights = np.broadcast_to(wavelengths.weights, sh) * wavelengths.weights.unit

    for w in range(wavelengths.num_per_config):

        util.set_float(zemax_system, wavls[..., w], csh, op_wave, unit_wave, w)

        util.set_float(zemax_system, weights[..., w], csh, op_weight, unit_weight, w)

