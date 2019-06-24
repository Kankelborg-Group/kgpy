
import pytest
import os
from typing import List
import numpy as np
import astropy.units as u

from kgpy.optics import Surface, Component, ZmxSystem, Surface


@pytest.fixture(scope='class')
def components():
    # Define a stop component and surface
    stop = Component('Stop')
    stop.append(Surface('Stop'))

    # Define a lens component and surface
    lens = Component('Primary')
    lens.append(Surface('Primary'))

    # Define a detector component and surface
    img = Component('Detector')
    img.append(Surface('Detector'))

    # Define a list of components we want to use to build the zemax system
    components = [stop, lens, img]

    return components


@pytest.fixture(scope='class')
def zmx_system(components: List[Component]):
    # sys = ZmxSystem('Test System', model_path=TestZmxSystem.test_path)
    sys = ZmxSystem('Test System')

    return sys


class TestZmxSystem:

    # Define location of a test Zemax file
    test_path = os.path.join(os.path.dirname(__file__), 'test_model.zmx')

    def test__init__(self, zmx_system: ZmxSystem):

        # Check that the Zemax instance started correctly
        assert zmx_system.example_constants() is not None

        # Check that all the surfaces are defined
        for surface in zmx_system:
            assert isinstance(surface, Surface)

        # Check the object surface
        assert zmx_system.object.component.name == ZmxSystem.object_str

        # Check the stop surface
        assert zmx_system.stop.component.name == ZmxSystem.main_str

        # Check the image surface
        assert zmx_system.image.component.name == ZmxSystem.main_str

    def test_from_file(self):

        sys = ZmxSystem.from_file('test', self.test_path)

        assert sys.__str__() is not None

    def test_raytrace(self, zmx_system: ZmxSystem):

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
        V, X, Y = zmx_system.raytrace(surf_indices, wavl_indices, fx, fy, px, py)

        # Check that the ray was not vignetted
        assert V[0] == 0

        # Check that the ray traversed the expected amount in the y-direction
        assert np.isclose(X[0], 0)
        assert np.isclose(Y[0], 100 * np.sin(np.radians(1.0)), rtol=1e-3)

    def test_read_uda_file(self):

        uda_file = os.path.join(os.path.dirname(__file__), 'test.uda')

        aper = ZmxSystem._read_uda_file(uda_file)

        x, y = aper.exterior.xy

        assert x[0] == -1.91
        assert y[3] == 8.49

    def test_find_surface(self, zmx_system: ZmxSystem):

        # Check that we can locate the primary surface
        primary_comment = 'Primary'
        true_primary_ind = 3
        primary_surf = zmx_system._find_surface(primary_comment)
        assert primary_surf.SurfaceNumber == true_primary_ind

        # Check that a comment matching none of the surfaces returns no surface
        unmatching_comment = 'asdfasdf'
        surf = zmx_system._find_surface(unmatching_comment)
        assert surf is None

    @pytest.mark.parametrize('s, tok', [
        ('Dummy',               (None, 'Dummy', None)),
        ('Dummy.Dummy', ('Dummy', 'Dummy', None)),
        ('Spider.Front',        ('Spider', 'Front', None)),
        ('Primary.tilt_dec',    (None, 'Primary', 'tilt_dec')),
        ('Baffle1.Pass1.aper',  ('Baffle1', 'Pass1', 'aper')),
    ])
    def test_parse_comment(self, s, tok):

        # Check that the output from the comment parsing is as expected
        assert tok == ZmxSystem._parse_comment(s)

    @pytest.mark.parametrize('cs, is_camel', [
        ("camel",       False),
        ("camelCase",   True),
        ("CamelCase",   True),
        ("CAMELCASE",   False),
        ("camelcase",   False),
        ("Camelcase",   True),
        ("Case",        True),
        ("CAMELcase",   True),
    ])
    def test_is_camel_case(self, cs, is_camel):
        assert ZmxSystem.is_camel_case(cs) is is_camel

    def test_lens_units(self, zmx_system: ZmxSystem):

        # Check that Zemax uses millimeters as default
        assert zmx_system.lens_units == u.mm

        # Check inches
        zmx_system.lens_units = u.imperial.inch
        assert zmx_system.lens_units == u.imperial.inch

        # Check centimeters
        zmx_system.lens_units = u.cm
        assert zmx_system.lens_units == u.cm

        # Check meters
        zmx_system.lens_units = u.m
        assert zmx_system.lens_units == u.m







