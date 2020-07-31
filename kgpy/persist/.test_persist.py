
from pathlib import Path

import kgpy
from .example_package import ExamplePackage
from . import persist
# from kgpy.optics import ZmxSystem/


def test_persist():

    f = persist.persist(ExamplePackage.f)

    f(1, 2, a=3)



# class TestPersist:
#
#     def test__init__(self):
#
#         val = TestPackage()
#
#     def test_get_imports(self):
#
#         imports = ZmxSystem.get_pydeps(pkg_list=Path(kgpy.__file__).parent)
#
#         for im in imports:
#             print(im)
