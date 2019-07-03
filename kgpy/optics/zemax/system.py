
import win32com
import win32com.client.gencache
import astropy.units as u

from kgpy import optics

from . import ZOSAPI

field_op_types = [
    ZOSAPI.Editors.MCE.MultiConfigOperandType.XFIE,
    ZOSAPI.Editors.MCE.MultiConfigOperandType.YFIE,
    ZOSAPI.Editors.MCE.MultiConfigOperandType.FVCX,
    ZOSAPI.Editors.MCE.MultiConfigOperandType.FVCY,
    ZOSAPI.Editors.MCE.MultiConfigOperandType.FVDX,
    ZOSAPI.Editors.MCE.MultiConfigOperandType.FVDY,
    ZOSAPI.Editors.MCE.MultiConfigOperandType.FVAN
]

def calc_zemax_system(system: optics.System) -> ZOSAPI.IOpticalSystem:

    # Create COM connection to Zemax
    zemax_connection = win32com.client.gencache.EnsureDispatch(
        "ZOSAPI.ZOSAPI_Connection")  # type: ZOSAPI.ZOSAPI_Connection

    # Open Zemax system
    zemax_system = zemax_connection.CreateNewApplication().PrimarySystem

    zemax_system.SystemData.Units = ZOSAPI.SystemData.ZemaxSystemUnits.Millimeters
    zemax_units = u.mm

    mce = zemax_system.MCE

    for configuration_index, configuration in enumerate(system.configurations):

        if configuration_index == 0:

            # op = mce.AddOperand()
            # op.Type = ZOSAPI.Editors.MCE.MultiConfigOperandType.SATP

            op = mce.AddOperand()
            op.Type = ZOSAPI.Editors.MCE.MultiConfigOperandType.APER

        else:

            mce.AddConfiguration(False)

        mce.SetCurrentConfiguration(configuration_index + 1)

        zemax_system.SystemData.Aperture.ApertureType = ZOSAPI.SystemData.ZemaxApertureType.EntrancePuilDiameter
        zemax_system.SystemData.Aperture.ApertureValue = (2 * configuration.entrance_pupil_radius).to(zemax_units).value



        for field_index, field in enumerate(configuration.fields):

            if configuration_index == 0:

                for op_type in field_op_types:

                    op = mce.AddOperand()
                    op.Type = op_type
                    op.Param1 = field_index


        for surface_index, surface in enumerate(configuration.surfaces):
            pass