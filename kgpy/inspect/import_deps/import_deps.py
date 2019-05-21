
import inspect
import typing


def get_imports(object, filter_path=None):

    if inspect.isclass(object):

        module = inspect.getmodule(object)

        members = inspect.getmembers(module)

        for member in members:

            print(member)

def get_class_imports(cls, filter_path=None):

    pass