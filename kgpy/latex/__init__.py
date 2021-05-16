import dataclasses
import pylatex

__all__ = ['Title', 'Author', 'Abstract', 'aas']


@dataclasses.dataclass
class Title(pylatex.base_classes.LatexObject):
    name: str

    def dumps(self):
        return pylatex.Command('title', self.name).dumps()


@dataclasses.dataclass
class Author(pylatex.base_classes.LatexObject):
    name: str

    def dumps(self):
        return pylatex.Command('author', self.name).dumps()


class Abstract(pylatex.base_classes.Environment):
    pass


from . import aas
