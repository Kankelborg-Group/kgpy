
import inspect
import typing


def get_imports(obj, filter_path=None):

    if inspect.isclass(obj):

        get_class_imports(obj, filter_path=filter_path)


def get_class_imports(cls, filter_path=None):

    cls_module = inspect.getmodule(cls)

    classes = inspect.getmembers(cls_module, inspect.isclass)
    modules = inspect.getmembers(cls_module, inspect.ismodule)
    functions = inspect.getmembers(cls_module, inspect.isfunction)

    members = classes + modules + functions

    for m in members:
        print(m)
        get_imports(m)
