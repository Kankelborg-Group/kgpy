
import abc
import typing as tp
import ast
import inspect
import pickle
import importlib
import importlib.util
import pathlib
import hashlib
import types
import klepto


def persist(func: tp.Callable, pkg_filter_list: tp.Optional[tp.List] = None, ):

    def persist_wrapper(*args, **kwargs):

        module = inspect.getmodule(func)

        arg_cache = klepto.archives.file_archive(name=cache_directory(module, func) / 'args',
                                                 serialized=True, cached=True)

        for arg in args:
            pass

            




        # if obj_path(p).exists():
        #     if cls.pydeps_path(p).exists():
        #         if cls.pydeps_unchanged(p, module=module, pkg_list=pkg_list):
        #             return cls.load(p)
        #
        # self = type.__call__(cls, *args, **kwargs)
        #
        # cls.save(self, p)
        #
        # return self

    return persist_wrapper



def __call__(cls, *args, **kwargs):

    module = inspect.getmodule(cls)

    pkg_list = cls.pkg_filter_list()

    path = cls.path(cls.__name__, module)

    if cls.obj_path(path).exists():
        if cls.pydeps_path(path).exists():
            if cls.pydeps_unchanged(path, module=module, pkg_list=pkg_list):
                return cls.load(path)

    self = type.__call__(cls, *args, **kwargs)

    cls.save(self, path)

    return self


def name(module: types.ModuleType):
    return module.__name__


def load(mcs, path: pathlib.Path):
    with mcs.obj_path(path).open(mode='rb') as f:
        return pickle.load(f)


def save(mcs, self, path: pathlib.Path):
    with mcs.obj_path(path).open(mode='wb') as f:
        pickle.dump(self, f, 0)


def obj_path(mcs, path: pathlib.Path) -> pathlib.Path:
    return path / pathlib.Path('obj')


def pydeps_path(mcs, path: pathlib.Path) -> pathlib.Path:
    return path / pathlib.Path('pydeps')

def args_path(mcs, path: pathlib.Path) -> pathlib.Path:
    return path / pathlib.Path('args')

def kwargs_path(mcs, path: pathlib.Path) -> pathlib.Path:
    return path / pathlib.Path('kwargs')


def cache_directory(module: types.ModuleType, func: tp.Callable) -> pathlib.Path:

    module_path = pathlib.Path(module.__file__).parent
    cache_path = pathlib.Path('.cache')
    module_name = pathlib.Path(module.__name__.split('.')[-1])
    func_name = pathlib.Path(func.__name__)
    path = module_path / cache_path / module_name / func_name
    path.mkdir(parents=True, exist_ok=True)

    return path


def pydeps_unchanged(mcs, path: pathlib.Path, module=None, pkg_list: tp.List = None):

    deps = mcs.get_pydeps(module=module, pkg_list=pkg_list)
    new_hashes = mcs.hash_pydeps(deps)

    # print(deps)

    try:
        # with open(cls.pydeps_path(name), 'rb') as f:
        with mcs.pydeps_path(path).open(mode='rb') as f:
            old_hashes = pickle.load(f)

        ret = new_hashes == old_hashes

    except FileNotFoundError:
        ret = False

    with mcs.pydeps_path(path).open(mode='wb') as f:
        pickle.dump(new_hashes, f)

    return ret


def hash_pydeps(deps: tp.List[str]):

    hashes = []

    for im in deps:

        with open(im, 'rb') as f:

            hash = hashlib.md5(f.read()).hexdigest()

            hashes.append(hash)

    return hashes


def get_pydeps(mcs, module=None, deps=None, pkg_list: tp.List = None):

    if deps is None:
        deps = []

    if module is None:
        module = inspect.getmodule(mcs)

    if not hasattr(module, '__file__'):
        return deps

    path = pathlib.Path(module.__file__)

    pkg_found = False
    if pkg_list is not None:
        for pkg in pkg_list:
            pkg_path = pathlib.Path(pkg.__file__).parent
            if pkg_path in path.parents:
                pkg_found=True

    if not pkg_found:
        return deps

    if path in deps:
        return deps

    deps.append(path)

    with path.open(mode='rb') as fh:
        root = ast.parse(fh.read(), path)

    module_globals = {
        '__name__': module.__name__,
        '__package__': module.__package__,
    }

    for node in ast.walk(root):

        if isinstance(node, ast.Import):

            for alias in node.names:

                new_module = importlib.__import__(alias.name, globals=module_globals, fromlist=[], level=0)

                deps = mcs.get_pydeps(new_module, deps, pkg_list=pkg_list)

        elif isinstance(node, ast.ImportFrom):

            name = node.module

            if name is None:
                name = ''

            for alias in node.names:

                new_module = importlib.__import__(name, globals=module_globals, fromlist=[alias.name], level=node.level)

                deps = mcs.get_pydeps(new_module, deps, pkg_list=pkg_list)

        else:
            continue

    return deps

