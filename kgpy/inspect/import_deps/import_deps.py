
from os import path
from pathlib import Path
import sys
import types
import inspect
import typing as t


def get_imports(test_obj: t.Any, node_list: t.List, module_list: t.List, lvl, filter_package: types.ModuleType = None):

    print(test_obj)

    node_list.append(test_obj)

    # Find the module associated with the test object
    mod = inspect.getmodule(test_obj)

    if not hasattr(mod, '__file__'):
        return node_list, module_list

    if filter_package is not None:
        if Path(filter_package.__file__).parent not in Path(mod.__file__).parents:
            return node_list, module_list

    # If we haven't seen this module yet, add it to the list
    if mod not in module_list:
        module_list.append(mod)
        node_list.append(mod)

    classes = inspect.getmembers(mod, inspect.isclass)
    modules = inspect.getmembers(mod, inspect.ismodule)
    functions = inspect.getmembers(mod, inspect.isfunction)

    classes += inspect.getmembers(test_obj, inspect.isclass)
    modules += inspect.getmembers(test_obj, inspect.ismodule)
    functions += inspect.getmembers(test_obj, inspect.isfunction)
    # functions = []

    members = classes + modules + functions

    # members = inspect.getmembers(test_obj)
    #
    # members += inspect.getmembers(mod)

    for _, m in members:

        if m == type:
            continue

        # if isinstance(m, types.FunctionType):
        #     continue

        if m in node_list:
            continue

        try:
            node_list, module_list = get_imports(m, node_list, module_list, lvl + 1, filter_package=filter_package)
        except:
            pass

    return node_list, module_list
