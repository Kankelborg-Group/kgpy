import typing as tp
import astropy.units as u

from kgpy import optics
from kgpy.optics.zemax import ZOSAPI

from . import util


def add_to_zemax_system(
        zemax_system: ZOSAPI.IOpticalSystem,
        fields: 'optics.system.Fields',
        configuration_shape: tp.Tuple[int],
):
    
    while zemax_system.SystemData.Fields.NumberOfFields < fields.num_per_config:
        zemax_system.SystemData.Fields.AddField(0, 0, 1)
        
    op_x = ZOSAPI.Editors.MCE.MultiConfigOperandType.XFIE
    op_y = ZOSAPI.Editors.MCE.MultiConfigOperandType.YFIE
    op_vcx = ZOSAPI.Editors.MCE.MultiConfigOperandType.FVCX
    op_vcy = ZOSAPI.Editors.MCE.MultiConfigOperandType.FVCY
    op_vdx = ZOSAPI.Editors.MCE.MultiConfigOperandType.FVDX
    op_vdy = ZOSAPI.Editors.MCE.MultiConfigOperandType.FVDY
    op_van = ZOSAPI.Editors.MCE.MultiConfigOperandType.FVAN
    
    unit_x = u.deg
    unit_y = u.deg
    unit_vdx = u.dimensionless_unscaled
    unit_vdy = u.dimensionless_unscaled
    unit_vcx = u.dimensionless_unscaled
    unit_vcy = u.dimensionless_unscaled
    unit_van = u.dimensionless_unscaled
    
    for f in range(fields.num_per_config):
        
        field_index = f + 1
        
        util.set_float(zemax_system, fields.x[..., f], configuration_shape, op_x, unit_x, param_1=field_index)
        util.set_float(zemax_system, fields.y[..., f], configuration_shape, op_y, unit_y, param_1=field_index)
        util.set_float(zemax_system, fields.vdx[..., f], configuration_shape, op_vdx, unit_vdx, param_1=field_index)
        util.set_float(zemax_system, fields.vdy[..., f], configuration_shape, op_vdy, unit_vdy, param_1=field_index)
        util.set_float(zemax_system, fields.vcx[..., f], configuration_shape, op_vcx, unit_vcx, param_1=field_index)
        util.set_float(zemax_system, fields.vcy[..., f], configuration_shape, op_vcy, unit_vcy, param_1=field_index)
        util.set_float(zemax_system, fields.van[..., f], configuration_shape, op_van, unit_van, param_1=field_index)

