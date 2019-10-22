from . import CoordinateBreak


class TestCoordinateBreak:

    def test__init__(self):

        c = CoordinateBreak()

        assert c.decenter.ndim == 3
        assert c.decenter.shape[~0] == 3

