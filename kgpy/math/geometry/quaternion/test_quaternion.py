
import pytest
from numpy import pi, isclose

from .quaternion import *

angles = [0, pi/3, -pi / 3]


@pytest.mark.parametrize('a', angles)
@pytest.mark.parametrize('b', angles)
@pytest.mark.parametrize('c', angles)
def test_xyz_intrinsic_tait_bryan_angles(a, b, c):

    print(a, b, c)

    q = from_zyx_intrinsic_tait_bryan_angles(a, b, c)

    a1, b1, c1 = as_zyx_intrinsic_tait_bryan_angles(q)

    assert isclose(a, a1)
    assert isclose(b, b1)
    assert isclose(c, c1)