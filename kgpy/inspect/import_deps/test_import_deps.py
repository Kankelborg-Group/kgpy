
import importlib
import import_deps
import inspect
import findimports

from kgpy.inspect.import_deps import get_imports
from kgpy.inspect.import_deps import TestPackage

from kgpy.optics import System


def test_get_imports():

    # objects, modules = get_imports(TestPackage, [], [], 0, filter_package=importlib.import_module('kgpy'))
    # import_deps.__main__.main(inspect.getmodule(TestPackage).__file__)

    im = findimports.ImportFinder(inspect.getmodule(TestPackage).__file__)

    print(im)

    print('_____')

    # for m in modules:
    #     print(m)
