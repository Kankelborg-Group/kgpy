from . import DiffractionGrating


class TestDiffractionGrating:

    def test__init__(self):

        d = DiffractionGrating()

        assert d.diffraction_order.ndim == 2
        assert d.groove_frequency.ndim == 2
