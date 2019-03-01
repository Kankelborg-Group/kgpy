
from typing import Union, List
import numpy as np
from numpy import sin, cos
from quaternion import quaternion, as_quat_array, as_float_array

__all__ = ['from_xyz_intrinsic_tait_bryan_angles', 'as_xyz_intrinsic_tait_bryan_angles']


def from_xyz_intrinsic_tait_bryan_angles(alpha_beta_gamma: Union[float, List[float], np.ndarray],
                                         beta: Union[float, np.ndarray] = None, gamma: Union[float, np.ndarray] = None
                                         ) -> quaternion:
    """
    Based on the quaternion/from_euler_angles() function in numpy-quaternion.
    The conversions from Tait-Bryan angles were taken from this Wikipedia page:
    https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Euler_angles_(z-y%E2%80%B2-x%E2%80%B3_intrinsic)_%E2%86%92_quaternion

    :param alpha_beta_gamma: This argument may either contain an array with last dimension of size 3, where those three
    elements describe the (alpha, beta, gamma) radian values for each rotation; or it may contain just the alpha values,
    in which case the next two arguments must also be given.
    :param beta: If this array is given, it must be able to broadcast against the first and third arguments.
    :param gamma: If this array is given, it must be able to broadcast against the first and second arguments.
    :return: quaternion array. The shape of this array will be the same as the input, except that the last dimension
    will be removed.
    """

    # Figure out the input angles from either type of input
    if gamma is None:
        alpha_beta_gamma = np.asarray(alpha_beta_gamma, dtype=np.double)
        alpha = alpha_beta_gamma[..., 0]
        beta = alpha_beta_gamma[..., 1]
        gamma = alpha_beta_gamma[..., 2]
    else:
        alpha = np.asarray(alpha_beta_gamma, dtype=np.double)
        beta = np.asarray(beta, dtype=np.double)
        gamma = np.asarray(gamma, dtype=np.double)

    # Set up the output array
    R = np.empty(np.broadcast(alpha, beta, gamma).shape + (4,), dtype=np.double)

    # Compute the actual values of the quaternion components
    R[..., 0] = cos(alpha / 2) * cos(beta / 2) * cos(gamma / 2) + sin(alpha / 2) * sin(beta / 2) * sin(gamma / 2)  # r
    R[..., 1] = sin(alpha / 2) * cos(beta / 2) * cos(gamma / 2) - cos(alpha / 2) * sin(beta / 2) * sin(gamma / 2)  # x
    R[..., 2] = cos(alpha / 2) * sin(beta / 2) * cos(gamma / 2) + sin(alpha / 2) * cos(beta / 2) * sin(gamma / 2)  # y
    R[..., 3] = cos(alpha / 2) * cos(beta / 2) * sin(gamma / 2) - sin(alpha / 2) * sin(beta / 2) * cos(gamma / 2)  # z

    return as_quat_array(R)


def as_xyz_intrinsic_tait_bryan_angles(q: quaternion) -> np.ndarray:
    """
    Convert a quaternion to Tait-Bryan angles using the same conventions described in the function above.

    :param q: Quaterion to express in terms of Tait-Bryan angles
    :return:
    """

    # Set up the output array
    alpha_beta_gamma = np.empty(q.shape + (3,), dtype=np.float)

    # Convert quaternion to raw numpy array
    q = as_float_array(q)

    # Save shortcuts to the components of the quaternion
    q_r = q[..., 0]
    q_i = q[..., 1]
    q_j = q[..., 2]
    q_k = q[..., 3]

    print(q_r, q_i, q_j, q_k)

    # Calculate the Tait-Bryan angles from the quaternion components
    alpha_beta_gamma[..., 0] = np.arctan2(2 * (q_r * q_i + q_j * q_k), 1 - 2 * (q_i * q_i + q_j * q_j))
    alpha_beta_gamma[..., 1] = np.arcsin(2 * (q_r * q_j - q_k * q_i))
    alpha_beta_gamma[..., 2] = np.arctan2(2 * (q_r * q_k + q_i * q_j), 1 - 2 * (q_j * q_j + q_k * q_k))

    print(alpha_beta_gamma)

    return alpha_beta_gamma
