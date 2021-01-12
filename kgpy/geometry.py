import typing as typ
import numpy as np
import astropy.units as u
from kgpy import vector

__all__ = ['line_plane_intercept', 'segment_plane_intercept']


def line_plane_intercept_parameter(
        plane_point: u.Quantity,
        plane_normal: u.Quantity,
        line_point: u.Quantity,
        line_direction: u.Quantity,
) -> u.Quantity:

    a = vector.dot(plane_point - line_point, plane_normal)
    b = vector.dot(line_direction, plane_normal)

    return a / b


def line_plane_intercept(
        plane_point: u.Quantity,
        plane_normal: u.Quantity,
        line_point: u.Quantity,
        line_direction: u.Quantity,
) -> u.Quantity:

    d = line_plane_intercept_parameter(plane_point, plane_normal, line_point, line_direction)

    return line_point + line_direction * d


def segment_plane_intercept(
        plane_point: u.Quantity,
        plane_normal: u.Quantity,
        line_point_1: u.Quantity,
        line_point_2: u.Quantity,
) -> u.Quantity:

    line_direction = line_point_2 - line_point_1
    line_length = vector.length(line_direction)

    d = line_plane_intercept_parameter(plane_point, plane_normal, line_point_1, line_direction)[..., 0]
    intercept = line_plane_intercept(plane_point, plane_normal, line_point_1, line_direction)

    intercept[d < 0, :] = np.nan
    intercept[d > 1, :] = np.nan

    return intercept
