import typing as typ
import dataclasses
from kgpy import mixin
from . import Image

__all__ = ['CubeAxis', 'Cube', 'SliceAxis', 'Slice', ]


class CubeAxis(mixin.AutoAxis):
    """
    Representation of the data axes for the arrays within :class:`kgpy.obs.spectral.Cube`
    """
    def __init__(self):
        super().__init__()
        self.w = self.auto_axis_index()
        self.x = self.auto_axis_index()
        self.y = self.auto_axis_index()
        self.time = self.auto_axis_index(from_right=False)
        self.channel = self.auto_axis_index(from_right=False)

    @property
    def xy(self) -> typ.Tuple[int, int]:
        return self.x, self.y


@dataclasses.dataclass
class Cube(Image):
    """
    Represents a sequence of spectrally-resolved images (data with two spatial dimensions and one spectral dimension).
    """
    axis: typ.ClassVar[CubeAxis] = CubeAxis()

    @property
    def num_wavelength(self) -> int:
        return self.shape[self.axis.w]


class SliceAxis(mixin.AutoAxis):
    def __init__(self):
        super().__init__()
        self.w = self.auto_axis_index()
        self.y = self.auto_axis_index()
        self.time = self.auto_axis_index(from_right=False)
        self.channel = self.auto_axis_index(from_right=False)


@dataclasses.dataclass
class Slice(Image):
    """
    Represents of sequence of images with one spectral axis and one spatial axis.
    This is the type of data that is natively gathered by slit imaging spectrographs such as IRIS.
    """
    axis: typ.ClassVar[SliceAxis] = SliceAxis()
