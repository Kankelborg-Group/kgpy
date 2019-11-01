import typing as tp
import nptyping as npt
import win32com
import win32com.client.gencache
import astropy.units as u
from kgpy.optics.zemax import ZOSAPI
from kgpy import optics
from . import wavelengths, util, fields, surface




def calc_zemax_system(system: 'optics.System') -> ZOSAPI.IOpticalSystem:

    zemax_system = open_zemax_system()

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
    surface.add_surfaces_to_zemax_system(zemax_system, system.surfaces, configuration_shape, zemax_lens_units)

    set_entrance_pupil_radius(zemax_system, system.entrance_pupil_radius, configuration_shape, zemax_lens_units)

    return zemax_system


def open_zemax_system() -> ZOSAPI.IOpticalSystem:

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
    return zemax_system


def set_entrance_pupil_radius(
        zemax_system: ZOSAPI.IOpticalSystem,
        entrance_pupil_radius: u.Quantity,
        configuration_shape: tp.Tuple[int], 
        zemax_units: u.Unit
):
    zemax_system.SystemData.Aperture.ApertureType = ZOSAPI.SystemData.ZemaxApertureType.EntrancePuilDiameter

    op_type = ZOSAPI.Editors.MCE.MultiConfigOperandType.APER

    entrance_pupil_diameter = 2 * entrance_pupil_radius
    util.set_float(zemax_system, entrance_pupil_diameter, configuration_shape, op_type, zemax_unit=zemax_units)


def set_stop_surface(
        zemax_system: ZOSAPI.IOpticalSystem,
        stop_surface_index: tp.Union[int, npt.Array[int]],
        configuration_shape: tp.Tuple[int],
):

    op_stop_surface_index = ZOSAPI.Editors.MCE.MultiConfigOperandType.STPS
    util.set_int(zemax_system, stop_surface_index, configuration_shape, op_stop_surface_index)