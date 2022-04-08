import pytest
from . import validation


@pytest.mark.skip
def test_simple_emission_line():
    validation.simple_emission_line()