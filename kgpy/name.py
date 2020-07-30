import dataclasses
import typing as typ


@dataclasses.dataclass
class Name:
    """
    Representation of a hierarchical namespace.
    Names are a composition of a parent, which is also a name, and a base which is a simple string.
    The string representation of a name is <parent>.base, where <parent> is the parent's string expansion.
    """

    base: str = ''  #: Base of the name, this string will appear last in the string representation
    parent: 'typ.Optional[Name]' = None     #: Parent string of the name, this itself also a name

    def copy(self):
        return type(self)(
            base=self.base,
            parent=self.parent.copy(),
        )

    def __add__(self, other: str) -> 'Name':
        """
        Quickly create the name of a child's name by adding a string to the current instance.
        Adding a string to a name instance returns
        :param other: A string representing the basename of the new Name instance.
        :return: A new `kgpy.Name` instance with the `self` as the `parent` and `other` as the `base`.
        """
        return type(self)(base=other, parent=self)

    def __repr__(self):
        if self.parent is not None:
            return self.parent.__repr__() + '.' + self.base

        else:
            return self.base
