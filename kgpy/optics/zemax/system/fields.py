import typing as tp
import numpy as np
import astropy.units as u

from kgpy import optics
from kgpy.optics.zemax import ZOSAPI

from . import util


def add_to_zemax_system(
        zemax_system: ZOSAPI.IOpticalSystem,
        fields: 'optics.system.Fields',
        configuration_shape: tp.Tuple[int],
):
    zemax_system.SystemData.Fields.Normalization = ZOSAPI.SystemData.FieldNormalizationType.Rectangular

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
    unit_van = u.deg

    sh = configuration_shape + (fields.num_per_config,)

    x = np.broadcast_to(fields.x, sh) * fields.x.unit
    y = np.broadcast_to(fields.y, sh) * fields.y.unit
    vdx = np.broadcast_to(fields.vdx, sh) * fields.vdx.unit
    vdy = np.broadcast_to(fields.vdy, sh) * fields.vdy.unit
    vcx = np.broadcast_to(fields.vcx, sh) * fields.vcx.unit
    vcy = np.broadcast_to(fields.vcy, sh) * fields.vcy.unit
    van = np.broadcast_to(fields.van, sh) * fields.van.unit
    
    for f in range(fields.num_per_config):

        util.set_float(zemax_system, x[..., f], configuration_shape, op_x, unit_x, param_1=f)
        util.set_float(zemax_system, y[..., f], configuration_shape, op_y, unit_y, param_1=f)
        util.set_float(zemax_system, vdx[..., f], configuration_shape, op_vdx, unit_vdx, param_1=f)
        util.set_float(zemax_system, vdy[..., f], configuration_shape, op_vdy, unit_vdy, param_1=f)
        util.set_float(zemax_system, vcx[..., f], configuration_shape, op_vcx, unit_vcx, param_1=f)
        util.set_float(zemax_system, vcy[..., f], configuration_shape, op_vcy, unit_vcy, param_1=f)
        util.set_float(zemax_system, van[..., f], configuration_shape, op_van, unit_van, param_1=f)

