import typing as tp
import win32com
import win32com.client.gencache
import astropy.units as u

from kgpy import optics
from . import util, fields, wavelengths

from . import ZOSAPI
from .fields import add_to_zemax_system
from .wavelengths import add_to_zemax_system
from .surface import add_surfaces_to_zemax_system


def calc_zemax_system(system: 'optics.System') -> ZOSAPI.IOpticalSystem:
    # Create COM connection to Zemax
    zemax_connection = win32com.client.gencache.EnsureDispatch(
        'ZOSAPI.ZOSAPI_Connection')  # type: ZOSAPI.ZOSAPI_Connection
    if zemax_connection is None:
        raise ValueError('Unable to initialize COM connection to ZOSAPI')

    # Open Zemax system
    zemax_app = zemax_connection.CreateNewApplication()
    if zemax_app is None:
        raise ValueError('Unable to acquire ZOSAPI application')

    # Check if license is valid
    if not zemax_app.IsValidLicenseForAPI:
        raise ValueError('License is not valid for ZOSAPI use')

    zemax_system = zemax_app.PrimarySystem
    if zemax_system is None:
        raise ValueError('Unable to acquire Primary system')

    # zemax_system.SystemData.Units = ZOSAPI.SystemData.ZemaxSystemUnits.Millimeters
    zemax_lens_units = u.mm

    configuration_size = system.config_broadcast.size
    configuration_shape = system.config_broadcast.shape

    op = zemax_system.MCE.AddOperand()
    op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.STPS)

    while zemax_system.MCE.NumberOfConfigurations < configuration_size:
        zemax_system.MCE.AddConfiguration(False)

    fields.add_to_zemax_system(zemax_system, system.fields, configuration_shape)
    wavelengths.add_to_zemax_system(zemax_system, system.wavelengths, configuration_shape)

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

        add_to_zemax_system(zemax_system, configuration_index, configuration.fields)
        add_to_zemax_system(zemax_system, configuration_index, configuration.wavelengths)
        add_surfaces_to_zemax_system(zemax_system, zemax_lens_units, configuration_index, configuration.surfaces)

    return zemax_system


def set_entrance_pupil_radius(
        zemax_system: ZOSAPI.IOpticalSystem,
        entrance_pupil_radius: u.Quantity,
        configuration_shape: tp.Tuple[int], 
        lens_units: u.Unit
):
    zemax_system.SystemData.Aperture.ApertureType = ZOSAPI.SystemData.ZemaxApertureType.EntrancePuilDiameter

    op_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.APER

    entrance_pupil_diameter = 2 * entrance_pupil_radius
    util.set_float(zemax_system, entrance_pupil_diameter, configuration_shape, op_type, zemax_unit=lens_units)
