import numpy as np
import typing as typ

from .simple_mart import SimpleMART


class LGOFMART(SimpleMART):
    type_int: typ.ClassVar[int] = 1

    @staticmethod
    def channel_is_not_converged(goodness_of_fit: np.ndarray) -> bool:
        return np.any(goodness_of_fit > 1)

        # return np.percentile(goodness_of_fit, 99.999) > 1

    @staticmethod
    def correction_exponent(goodness_of_fit: np.ndarray) -> np.ndarray:
        return goodness_of_fit > 1

