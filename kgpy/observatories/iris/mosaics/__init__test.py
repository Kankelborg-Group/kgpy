from . import load_index


def test_load_index(capsys):
    with capsys.disabled():
        load_index()