
from pathlib import Path

import kgpy
from . import Persist
from .test_package import TestPackage
from kgpy.optics import ZmxSystem

class TestPersist:

    def test__init__(self):

        val = TestPackage()

    def test_get_imports(self):

        imports = ZmxSystem.get_pydeps(filter_path=Path(kgpy.__file__).parent)

        for im in imports:
            print(im)
