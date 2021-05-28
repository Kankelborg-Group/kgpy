import typing as typ
import dataclasses
import pylatex
import astropy.units as u
import kgpy.format

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


from . import aas
