
import typing as t
import os
import ast
import inspect
import pickle
import importlib
import importlib.util
import pathlib as pl
import hashlib


class Persist:

    def __new__(cls, *args, **kwargs):

        name = args[0]
        filter_path = kwargs['filter_path']

        if cls.pydeps_unchanged(name, filter_path=filter_path):

            cls.__init__ = cls._init

            self = cls.load(name)


        else:

            self = super().__new__(cls)

            self.save(name)

        return self

    def _init(self, *args, **kwargs):
        pass

    @classmethod
    def load(cls, name: str):

        with cls.obj_path(name).open(mode='rb') as f:
            return pickle.load(f)

    def save(self, name: str):
        with self.obj_path(name).open(mode='wb') as f:
            pickle.dump(self, f, 0)

    @staticmethod
    def obj_path(name: str) -> pl.Path:
        return Persist.file_to_path(name + '.obj.pickle')

    @staticmethod
    def pydeps_path(name: str) -> pl.Path:
        return Persist.file_to_path(name + '.pydep.pickle')

    @staticmethod
    def args_path(name: str) -> pl.Path:
        return Persist.file_to_path(name + '.args.pickle')

    @staticmethod
    def kwargs_path(name: str) -> pl.Path:
        return Persist.file_to_path(name + '.kwargs.pickle')

    @staticmethod
    def file_to_path(file: str) -> pl.Path:

        file = pl.Path(file)
        path = pl.Path(__file__).parent
        return path / file

    @classmethod
    def pydeps_unchanged(cls, name: str, filter_path: t.Union[pl.Path, None] = None):

        deps = cls.get_pydeps(filter_path=filter_path)
        new_hashes = cls.hash_pydeps(deps)

        try:
            # with open(cls.pydeps_path(name), 'rb') as f:
            with cls.pydeps_path(name).open(mode='rb') as f:
                print(f)
                old_hashes = pickle.load(f)

            ret = new_hashes == old_hashes

        except FileNotFoundError:
            ret = False

        with cls.pydeps_path(name).open(mode='wb') as f:
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
    def get_pydeps(cls, module=None, deps=None, filter_path: t.Union[pl.Path, None] = None):

        if deps is None:
            deps = []

        if module is None:
            module = inspect.getmodule(cls)

        if not hasattr(module, '__file__'):
            return deps

        path = pl.Path(module.__file__)

        if filter_path is not None:
            if filter_path not in path.parents:
                return deps

        if path in deps:
            return deps

        deps.append(path)

        with open(path) as fh:
            root = ast.parse(fh.read(), path)

        module_globals = {
            '__name__': module.__name__,
            '__package__': module.__package__,
        }

        for node in ast.walk(root):

            if isinstance(node, ast.Import):

                for alias in node.names:

                    new_module = importlib.__import__(alias.name, globals=module_globals, fromlist=[], level=0)

                    deps = cls.get_pydeps(new_module, deps, filter_path=filter_path)

            elif isinstance(node, ast.ImportFrom):

                name = node.module

                if name is None:
                    name = ''

                for alias in node.names:

                    new_module = importlib.__import__(name, globals=module_globals, fromlist=[alias.name], level=node.level)

                    deps = cls.get_pydeps(new_module, deps, filter_path=filter_path)

            else:
                continue

        return deps

