import pytest
import numpy as np
import astropy.units as u
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vectors
from . import secant


@pytest.mark.parametrize(
    argnames='root_x',
    argvalues=[
        4,
        kgpy.labeled.LinearSpace(4, 6, 3, axis='xx'),
    ]
)
@pytest.mark.parametrize(
    argnames='root_y',
    argvalues=[
        None,
        7,
        kgpy.labeled.LinearSpace(7, 9, 3, axis='yy'),
    ]
)
@pytest.mark.parametrize(
    argnames='width_x',
    argvalues=[None, 1]
)
@pytest.mark.parametrize(
    argnames='width_y',
    argvalues=[None, 1]
)
@pytest.mark.parametrize(
    argnames='unit',
    argvalues=[1, u.mm]
)
def test_secant(
        root_x: kgpy.labeled.ArrayLike,
        root_y: kgpy.labeled.ArrayLike,
        width_x: kgpy.labeled.ArrayLike,
        width_y: kgpy.labeled.ArrayLike,
        unit: u.Unit
):
    if width_x is not None:
        root_x = kgpy.uncertainty.Uniform(root_x, width=width_x)

    if width_y is not None:
        if root_y is not None:
            root_y = kgpy.uncertainty.Uniform(root_y, width=width_y)

    if root_y is None:
        root_1 = root_x
    else:
        root_1 = kgpy.vectors.Cartesian2D(root_x, root_y)
    root_1 = root_1 * unit
    root_2 = 20 * unit

    def func(position: kgpy.labeled.AbstractArray) -> kgpy.labeled.AbstractArray:
        return (position - root_1) * (position - root_2)

    root_guess = 0 * root_1
    root = secant(
        func=func,
        root_guess=root_guess,
        step_size=0.1 * unit + root_guess,
        max_abs_error=1e-9 * unit * unit,
    )

    assert np.all(np.abs(root - root_1) < 1e-3 * unit)
