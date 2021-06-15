import pathlib
import typing as typ
import dataclasses
import pylatex
from . import Author as AuthorBase, textwidth

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


@dataclasses.dataclass
class Fig(pylatex.base_classes.LatexObject):
    _command_str: typ.ClassVar[str] = 'fig'
    file: pathlib.Path
    width: float
    caption: str

    def dumps(self) -> str:
        return pylatex.Command(self._command_str, arguments=[pylatex.NoEscape(self.file.as_posix()), self.width, self.caption]).dumps()


@dataclasses.dataclass
class LeftFig(Fig):
    _command_str: typ.ClassVar[str] = 'leftfig'


@dataclasses.dataclass
class RightFig(Fig):
    _command_str: typ.ClassVar[str] = 'rightfig'


@dataclasses.dataclass
class Gridline(pylatex.base_classes.LatexObject):
    figures: typ.List[Fig]

    def dumps(self) -> str:
        return pylatex.Command('gridline', arguments=pylatex.NoEscape('\n'.join([f.dumps() for f in self.figures]))).dumps()


