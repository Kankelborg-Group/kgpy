from .. import vector


def test_dot():

    assert vector.dot(vector.x_hat, vector.y_hat) == 0
    assert vector.dot(vector.x_hat, vector.x_hat) == 1
