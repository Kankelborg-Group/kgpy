import typing as tp
import nptyping as npt
import numpy as np
from astropy import units as u

from kgpy.optics.zemax import ZOSAPI


def set_float(
        zemax_system: ZOSAPI.IOpticalSystem,
        value: tp.Union[float, npt.Array[np.float], u.Quantity],
        configuration_shape: tp.Tuple[int],
        op_type: ZOSAPI.Editors.MCE.MultiConfigOperandType,
        zemax_unit: tp.Optional[u.Unit] = None,
        param_1: tp.Optional[int] = None,
        param_2: tp.Optional[int] = None,
        param_3: tp.Optional[int] = None,
):
    
    if zemax_unit is not None:
        value = value.to(zemax_unit).value

    value_broadcasted = np.broadcast_to(value, configuration_shape).flat

    op = zemax_system.MCE.AddOperand()
    op.ChangeType(op_type)
    
    if param_1 is not None:
        op.Param1 = param_1
        
    if param_2 is not None:
        op.Param2 = param_2
        
    if param_3 is not None:
        op.Param3 = param_3

    for i, value in value_broadcasted:
        config_index = i + 1
        cell = op.GetOperandCell(config_index)
        cell.DoubleValue = value


def set_str(
        zemax_system: ZOSAPI.IOpticalSystem,
        value: tp.Union[str, npt.Array[str]],
        configuration_shape: tp.Tuple[int],
        op_type: ZOSAPI.Editors.MCE.MultiConfigOperandType,
        param_1: tp.Optional[int] = None,
        param_2: tp.Optional[int] = None,
        param_3: tp.Optional[int] = None,
):
    value_broadcasted = np.broadcast_to(value, configuration_shape).flat

    op = zemax_system.MCE.AddOperand()
    op.ChangeType(op_type)

    if param_1 is not None:
        op.Param1 = param_1

    if param_2 is not None:
        op.Param2 = param_2

    if param_3 is not None:
        op.Param3 = param_3
    
    for i, value in value_broadcasted:
        config_index = i + 1
        cell = op.GetOperandCell(config_index)
        cell.Value = value