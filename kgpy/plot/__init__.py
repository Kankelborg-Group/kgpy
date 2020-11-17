
__all__ = ['ImageSlicer', 'CubeSlicer', 'HypercubeSlicer']

from ._image_slicer import ImageSlicer
from ._cube_slicer import CubeSlicer
from ._hypercube_slicer import HypercubeSlicer

ImageSlicer.__module__ = __name__
CubeSlicer.__module__ = __name__
HypercubeSlicer.__module__ = __name__
