import dataclasses
import numpy as np
import typing as typ

__all__ = ['Result']


@dataclasses.dataclass
class Result:
    cube: 'typ.Optional[np.ndarray]' = None
    best_cube: 'typ.Optional[np.ndarray]' = None
    best_filtering_iteration: typ.Optional[int] = None
    norm_history: typ.List[float] = dataclasses.field(default_factory=lambda: [])
    chisq_history: typ.List[float] = dataclasses.field(default_factory=lambda: [])
    mart_type_history: typ.List[int] = dataclasses.field(default_factory=lambda: [])
    cube_history: typ.List[np.ndarray] = dataclasses.field(default_factory=lambda: [])
    total_intensity_history: typ.List[np.ndarray] = dataclasses.field(default_factory=lambda: [])
    object_parameters: typ.Dict[str, typ.Any] = dataclasses.field(default_factory=lambda: {})
    call_parameters: typ.Dict[str, typ.Any] = dataclasses.field(default_factory=lambda: {})


