
import pytest
import os

from kgpy.optics import ZmxSystem
# noinspection PyUnresolvedReferences
from kgpy.optics.zemax.test_system import sys, components   # Import fixture


class TestSurface:
    test_path = os.path.join(os.path.dirname(__file__), 'test_model.zmx')

    def test_system_index(self, sys: ZmxSystem):

        # Check the index of the Primary surface
        primary_comment = 'Primary'
        true_primary_ind = 3
        primary_surf = sys._find_surface(primary_comment)

        print(primary_surf)
        assert primary_surf.SurfaceNumber == true_primary_ind

    def test_cs_break(self):

        pass