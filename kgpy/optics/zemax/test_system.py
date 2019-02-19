
import pytest
import math
import os
from typing import List
import numpy as np

from kgpy.optics import Surface, Component, ZmxSystem


@pytest.fixture
def components():
    # Define a stop component and surface
    stop = Component('Stop')
    stop.append_surface(Surface('Stop'))

    # Define a lens component and surface
    lens = Component('Primary')
    lens.append_surface(Surface('Primary'))

    # Define a detector component and surface
    img = Component('Detector')
    img.append_surface(Surface('Detector'))

    # Define a list of components we want to use to build the zemax system
    components = [stop, lens, img]

    return components


@pytest.fixture
def sys(components: List[Component]):
    sys = ZmxSystem('Test System', components=components, model_path=TestZmxSystem.test_path)

    return sys


class TestZmxSystem:

    test_path = os.path.join(os.path.dirname(__file__), 'test_model.zmx')

    def test__init__(self, components: List[Component], sys: ZmxSystem):

        # Check that the Zemax instance started correctly
        assert sys.example_constants() is not None

        # Check that the zmx_surf field contains a valid object for every surface in every component
        for orig_comp, zmx_comp in zip(components, sys.components):
            for orig_surf, zmx_surf in zip(orig_comp.surfaces, zmx_comp.surfaces):

                # Check that all comment strings are equal
                assert orig_surf.comment == zmx_surf.comment

    def test_raytrace(self, sys: ZmxSystem):

        # Surface we want to raytrace to is the last surface
        surf_indices = [-1]

        # User the first (and only) wavelength in the system
        wavl_indices = [1]

        # Choose a ray starting at the center of the pupil, but at an extreme field angle.
        fx = [0.0]
        fy = [1.0]
        px = [0.0]
        py = [0.0]

        # Execute the test raytrace
        V, X, Y = sys.raytrace(surf_indices, wavl_indices, fx, fy, px, py)

        # Check that the ray was not vignetted
        assert V[0] == 0

        # Check that the ray traversed the expected amount in the y-direction
        assert np.isclose(X[0], 0)
        assert np.isclose(Y[0], 100 * np.sin(np.radians(1.0)), rtol=1e-3)

    def test_find_surface(self, sys: ZmxSystem):

        # Check that we can locate the primary surface
        primary_comment = 'Primary'
        true_primary_ind = 2
        primary_surf = sys.find_surface(primary_comment)
        assert primary_surf.SurfaceNumber == true_primary_ind

        # Check that a comment matching none of the surfaces returns no surface
        unmatching_comment = 'asdfasdf'
        surf = sys.find_surface(unmatching_comment)
        assert surf is None


