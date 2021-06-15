import typing as typ
import dataclasses
import pylatex
import astropy.units as u
import kgpy.format

__all__ = ['Title', 'Author', 'Abstract', 'aas']

textwidth = pylatex.NoEscape(r'\textwidth')
columnwidth = pylatex.NoEscape(r'\columnwidth')


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


class Document(pylatex.Document):

    def set_variable_quantity(
            self,
            name: str,
            value: u.Quantity,
            scientific_notation: typ.Optional[bool] = None,
            digits_after_decimal: int = 3,
    ) -> typ.NoReturn:
        self.set_variable(
            name=name,
            value=pylatex.NoEscape(kgpy.format.quantity(
                a=value,
                scientific_notation=scientific_notation,
                digits_after_decimal=digits_after_decimal,
            ))
        )


class FigureStar(pylatex.Figure):
    def __init__(self, *, position: str = None, **kwargs):
        super().__init__(position=position, **kwargs)
        self._latex_name = 'figure*'


@dataclasses.dataclass
class Acronym(pylatex.base_classes.LatexObject):
    acronym: str
    name_full: str
    name_short: typ.Optional[str] = None
    plural: bool = False

    def dumps(self):
        command = pylatex.Command(
            command='newacro',
            arguments=[self.acronym, pylatex.NoEscape(self.name_full)],
            options=self.name_short,
        ).dumps()
        command += pylatex.Command(
            command='newcommand',
            arguments=[pylatex.NoEscape('\\' + self.acronym), pylatex.NoEscape(r'\ac{' + self.acronym + '}')],
        ).dumps()
        if self.plural:
            command += pylatex.Command(
                command='newcommand',
                arguments=[pylatex.NoEscape('\\' + self.acronym + 's'), pylatex.NoEscape(r'\acp{' + self.acronym + '}')],
            ).dumps()
        return command


@dataclasses.dataclass
class Label(pylatex.base_classes.LatexObject):
    name: str

    def dumps(self):
        return pylatex.Command('label', self.name).dumps()


from . import aas
