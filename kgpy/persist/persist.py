
import abc
import typing as t
import ast
import inspect
import pickle
import importlib
import importlib.util
import pathlib as pl
import hashlib
import types

__all__ = ['Persist', 'PersistMeta']


class PersistMeta(abc.ABCMeta):

    def __call__(cls, name: str, *args, **kwargs):

        module = inspect.getmodule(cls)

        pkg_list = cls.pkg_filter_list()

        path = cls.path(name, module)

        if cls.obj_path(path).exists():
            if cls.pydeps_path(path).exists():
                if cls.pydeps_unchanged(path, module=module, pkg_list=pkg_list):
                    return cls.load(path)

        self = type.__call__(cls, name, *args, **kwargs)

        cls.save(self, path)

        return self

    @staticmethod
    def name(module: types.ModuleType):
        return module.__name__

    @abc.abstractmethod
    def pkg_filter_list(cls) -> t.List[types.ModuleType]:
        pass

    @classmethod
    def load(mcs, path: pl.Path):

        with mcs.obj_path(path).open(mode='rb') as f:
            return pickle.load(f)

    @classmethod
    def save(mcs, self, path: pl.Path):
        with mcs.obj_path(path).open(mode='wb') as f:
            pickle.dump(self, f, 0)

    @classmethod
    def obj_path(mcs, path: pl.Path) -> pl.Path:
        return path / pl.Path('obj')

    @classmethod
    def pydeps_path(mcs, path: pl.Path) -> pl.Path:
        return path / pl.Path('pydeps')

    @classmethod
    def args_path(mcs, path: pl.Path) -> pl.Path:
        return path / pl.Path('args')

    @classmethod
    def kwargs_path(mcs, path: pl.Path) -> pl.Path:
        return path / pl.Path('kwargs')

    @staticmethod
    def path(name: str, module: types.ModuleType) -> pl.Path:

        module_path = pl.Path(module.__file__).parent
        pickle_path = pl.Path('.pickle_cache')
        name_path = pl.Path(name)
        path = module_path / pickle_path / name_path
        path.mkdir(parents=True, exist_ok=True)

        return path

    @classmethod
    def pydeps_unchanged(mcs, path: pl.Path, module=None, pkg_list: t.List = None):

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


    @staticmethod
    def hash_pydeps(deps: t.List[str]):

        hashes = []

        for im in deps:

            with open(im, 'rb') as f:

                hash = hashlib.md5(f.read()).hexdigest()

                hashes.append(hash)

        return hashes

    @classmethod
    def get_pydeps(mcs, module=None, deps=None, pkg_list: t.List = None):

        if deps is None:
            deps = []

        if module is None:
            module = inspect.getmodule(mcs)

        if not hasattr(module, '__file__'):
            return deps

        path = pl.Path(module.__file__)

        pkg_found = False
        if pkg_list is not None:
            for pkg in pkg_list:
                pkg_path = pl.Path(pkg.__file__).parent
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


class Persist(metaclass=PersistMeta):

    pass
