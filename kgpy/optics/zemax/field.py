
import typing as tp
import astropy.units as u

from kgpy import optics
from kgpy.optics.zemax import ZOSAPI


def add_fields_to_zemax_system(zemax_system: ZOSAPI.IOpticalSystem,
                               configuration_index: int,
                               fields: 'tp.List[optics.system.configuration.Field]'):
    field_op_types = [
        ZOSAPI.Editors.MCE.MultiConfigOperandType.XFIE,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.YFIE,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.FVCX,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.FVCY,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.FVDX,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.FVDY,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.FVAN,
    ]

    for field_index, field in enumerate(fields):

        if configuration_index == 0:

            if field_index == 0:
                zemax_field = zemax_system.SystemData.Fields.GetField(field_index + 1)

            else:
                zemax_field = zemax_system.SystemData.Fields.AddField(0.0, 0.0, 1.0)

            for op_type in field_op_types:
                op = zemax_system.MCE.AddOperand()
                op.ChangeType(op_type)
                op.Param1 = field_index

        else:
            zemax_field = zemax_system.SystemData.Fields.GetField(field_index + 1)

        zemax_field.X = field.x.to(u.deg).value
        zemax_field.Y = field.y.to(u.deg).value
        zemax_field.Weight = float(field.weight)
        zemax_field.VDX = float(field.vdx)
        zemax_field.VDY = float(field.vdy)
        zemax_field.VCX = float(field.vcx)
        zemax_field.VCY = float(field.vcy)
        zemax_field.VAN = float(field.van.to(u.deg).value)
