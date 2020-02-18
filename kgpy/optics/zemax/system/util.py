import typing as typ
import nptyping as npt
import numpy as np
from astropy import units as u

from kgpy.optics.zemax import ZOSAPI
from .system import System
from .configuration import Operand


def set_float(
        zemax_system: ZOSAPI.IOpticalSystem,
        value: typ.Union[float, npt.Array[np.float], u.Quantity],
        configuration_shape: typ.Tuple[int],
        op_type: ZOSAPI.Editors.MCE.MultiConfigOperandType,
        zemax_unit: typ.Optional[u.Unit] = None,
        param_1: typ.Optional[int] = None,
        param_2: typ.Optional[int] = None,
        param_3: typ.Optional[int] = None,
):
    
    if zemax_unit is not None:
        value = value.to(zemax_unit).value

    value_broadcasted = np.broadcast_to(value, configuration_shape).flat

    op = zemax_system.MCE.AddOperand()
    op.ChangeType(op_type)

    set_params(op, param_1, param_2, param_3)

    for i, value in enumerate(value_broadcasted):
        config_index = i + 1
        cell = op.GetOperandCell(config_index)
        cell.DoubleValue = value.item()


def set_str(
        zemax_system: ZOSAPI.IOpticalSystem,
        value: typ.Union[str, npt.Array[str]],
        configuration_shape: typ.Tuple[int],
        op_type: ZOSAPI.Editors.MCE.MultiConfigOperandType,
        param_1: typ.Optional[int] = None,
        param_2: typ.Optional[int] = None,
        param_3: typ.Optional[int] = None,
):
    value_broadcasted = np.broadcast_to(value, configuration_shape).flat

    op = zemax_system.MCE.AddOperand()
    op.ChangeType(op_type)

    set_params(op, param_1, param_2, param_3)
    
    for i, value in enumerate(value_broadcasted):
        config_index = i + 1
        cell = op.GetOperandCell(config_index)
        cell.Value = value


def set_int(
        zemax_system: ZOSAPI.IOpticalSystem,
        value: typ.Union[int, npt.Array[int]],
        configuration_shape: typ.Tuple[int],
        op_type: ZOSAPI.Editors.MCE.MultiConfigOperandType,
        param_1: typ.Optional[int] = None,
        param_2: typ.Optional[int] = None,
        param_3: typ.Optional[int] = None,
):
    value_broadcasted = np.broadcast_to(value, configuration_shape).flat

    op = zemax_system.MCE.AddOperand()
    op.ChangeType(op_type)

    set_params(op, param_1, param_2, param_3)

    for i, value in enumerate(value_broadcasted):
        config_index = i + 1
        cell = op.GetOperandCell(config_index)
        cell.IntegerValue = value.item()


def set_params(op, param_1, param_2, param_3):
    if param_1 is not None:
        op.Param1 = param_1
    if param_2 is not None:
        op.Param2 = param_2
    if param_3 is not None:
        op.Param3 = param_3
