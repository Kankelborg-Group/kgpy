import dataclasses
import typing as typ


@dataclasses.dataclass
class Name:

    base: str = ''
    parent: 'typ.Optional[Name]' = None

    def __add__(self, other: str) -> 'Name':
        return type(self)(self, other)

    def __str__(self):
        if self.parent is not None:
            return self.parent.__str__() + '.' + self.base

        else:
            return self.base
