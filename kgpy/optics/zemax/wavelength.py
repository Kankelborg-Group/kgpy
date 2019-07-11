
import typing as tp
import astropy.units as u

from kgpy import optics
from kgpy.optics.zemax import ZOSAPI

zemax_wavelength_units = u.um


def add_wavelengths_to_zemax_system(zemax_system: ZOSAPI.IOpticalSystem,
                                    configuration_index: int,
                                    wavelengths: 'tp.List[optics.system.configuration.Wavelength]'):
    wavelength_op_types = [
        ZOSAPI.Editors.MCE.MultiConfigOperandType.WAVE,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.WLWT,
    ]

    for wavelength_index, wavelength in enumerate(wavelengths):

        if configuration_index == 0:

            if wavelength_index == 0:
                zemax_wavelength = zemax_system.SystemData.Wavelengths.GetWavelength(wavelength_index + 1)
            else:
                zemax_wavelength = zemax_system.SystemData.Wavelengths.AddWavelength(1.0, 1.0)

            for op_type in wavelength_op_types:
                op = zemax_system.MCE.AddOperand()
                op.ChangeType(op_type)
                op.Param1 = wavelength_index

        else:
            zemax_wavelength = zemax_system.SystemData.Wavelengths.GetWavelength(wavelength_index + 1)

        zemax_wavelength.Wavelength = wavelength.wavelength.to(zemax_wavelength_units).value
        zemax_wavelength.Weight = float(wavelength.weight)
