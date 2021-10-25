import numpy as np
from . import rebin, DataArray


def test_rebin():
    x = np.random.rand(2, 5, 7, 3)
    scales = (2, 2, 2, 2)
    new_x = rebin(x, scales)
    assert new_x.shape == tuple(x.shape * np.array(scales))


class TestDataArray:

    def test_get_item(self):

        shape = dict(x=5, y=6)
        shape_linear = list(shape.values())

        x, y = np.ix_(np.arange(shape['x']), np.linspace(0, 1, shape['y']))

        d = DataArray(
            np.random.random(shape_linear),
            grid=dict(
                x=x,
                y=y,
            )
        )

        assert d.get_item(x=0).shape == dict(y=shape['y'])
        assert d.get_item(y=~0).shape == dict(x=shape['x'])

        slice_test = slice(0, 3)
        assert d.get_item(x=slice_test).shape == dict(x=slice_test.stop, y=shape['y'],)
        assert d.get_item(y=slice_test).shape == dict(x=shape['x'], y=slice_test.stop,)
