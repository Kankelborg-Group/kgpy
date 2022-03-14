import typing as typ
import numpy as np
import astropy.units as u
import kgpy.vectors

__all__ = [
    'line_plane_intercept',
    'segment_plane_intercept',
]


def line_plane_intercept_parameter(
        plane_point: kgpy.vectors.Cartesian3D,
        plane_normal: kgpy.vectors.Cartesian3D,
        line_point: kgpy.vectors.Cartesian3D,
        line_direction: kgpy.vectors.Cartesian3D,
) -> u.Quantity:

    a = (plane_point - line_point) @ plane_normal
    b = line_direction @ plane_normal

    return a / b


def line_plane_intercept(
        plane_point: kgpy.vectors.Cartesian3D,
        plane_normal: kgpy.vectors.Cartesian3D,
        line_point: kgpy.vectors.Cartesian3D,
        line_direction: kgpy.vectors.Cartesian3D,
) -> kgpy.vectors.Cartesian3D:

    d = line_plane_intercept_parameter(plane_point, plane_normal, line_point, line_direction)

    return line_point + line_direction * d


def segment_plane_intercept(
        plane_point: kgpy.vectors.Cartesian3D,
        plane_normal: kgpy.vectors.Cartesian3D,
        line_point_1: kgpy.vectors.Cartesian3D,
        line_point_2: kgpy.vectors.Cartesian3D,
) -> kgpy.vectors.Cartesian3D:

    line_direction = line_point_2 - line_point_1

    d = line_plane_intercept_parameter(plane_point, plane_normal, line_point_1, line_direction)
    intercept = line_plane_intercept(plane_point, plane_normal, line_point_1, line_direction)
    # intercept = line_point_1 + line_direction * d

    intercept[d < 0] = np.nan
    intercept[d > 1] = np.nan

    return intercept
