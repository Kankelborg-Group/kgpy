import dataclasses
import typing as typ


@dataclasses.dataclass
class Name:

    base: str = ''
    parent: 'typ.Optional[Name]' = None

    def copy(self):
        return type(self)(
            base=self.base,
            parent=self.parent.copy(),
        )

    def __add__(self, other: str) -> 'Name':
        return type(self)(base=other, parent=self)

    def __repr__(self):
        if self.parent is not None:
            return self.parent.__repr__() + '.' + self.base

        else:
            return self.base
