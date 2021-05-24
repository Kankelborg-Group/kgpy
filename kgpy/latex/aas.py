import dataclasses
import pylatex
from . import Author as AuthorBase

__all__ = ['Affiliation']


@dataclasses.dataclass
class Affiliation(pylatex.base_classes.LatexObject):
    name: str

    def dumps(self):
        return pylatex.Command('affiliation', self.name).dumps()


@dataclasses.dataclass
class Author(AuthorBase):
    affilation: Affiliation

    def dumps(self):
        return super().dumps() + self.affilation.dumps()



