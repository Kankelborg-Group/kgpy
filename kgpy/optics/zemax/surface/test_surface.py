import os

# noinspection PyUnresolvedReferences
from kgpy.optics.zemax.system.test_system import system, components   # Import fixture


class TestSurface:
    test_path = os.path.join(os.path.dirname(__file__), 'test_model.zmx')

    def test_system_index(self, system):

        pass

    def test_cs_break(self):

        pass
