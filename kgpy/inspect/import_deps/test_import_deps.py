
from kgpy.inspect.import_deps import get_imports
from .test_package import B


def test_get_imports():

    get_imports(B)
