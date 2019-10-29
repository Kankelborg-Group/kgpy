
import win32com
import win32com.client.gencache
import astropy.units as u

from kgpy import optics

from . import ZOSAPI
from .field import add_fields_to_zemax_system
from .wavelength import add_wavelengths_to_zemax_system
from .surface import add_surfaces_to_zemax_system


def calc_zemax_system(system: 'optics.System') -> ZOSAPI.IOpticalSystem:

    # Create COM connection to Zemax
    zemax_connection = win32com.client.gencache.EnsureDispatch(
        'ZOSAPI.ZOSAPI_Connection')  # type: ZOSAPI.ZOSAPI_Connection

    # Open Zemax system
    zemax_app = zemax_connection.CreateNewApplication()
    zemax_system = zemax_app.PrimarySystem

    # zemax_system.SystemData.Units = ZOSAPI.SystemData.ZemaxSystemUnits.Millimeters
    zemax_lens_units = u.mm

    for configuration_index, configuration in enumerate(system.configurations):

        if configuration_index == 0:

            # op = mce.AddOperand()
            # op.Type = ZOSAPI.Editors.MCE.MultiConfigOperandType.SATP

            op = zemax_system.MCE.AddOperand()
            op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.APER)

            op = zemax_system.MCE.AddOperand()
            op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.STPS)

        else:

            zemax_system.MCE.AddConfiguration(False)

        zemax_system.MCE.SetCurrentConfiguration(configuration_index + 1)

        zemax_system.SystemData.Aperture.ApertureType = ZOSAPI.SystemData.ZemaxApertureType.EntrancePuilDiameter
        zemax_system.SystemData.Aperture.ApertureValue = (2 * configuration.entrance_pupil_radius).to(
            zemax_lens_units).value

        add_fields_to_zemax_system(zemax_system, configuration_index, configuration.fields)
        add_wavelengths_to_zemax_system(zemax_system, configuration_index, configuration.wavelengths)
        add_surfaces_to_zemax_system(zemax_system, zemax_lens_units, configuration_index, configuration.surfaces)

    return zemax_system






