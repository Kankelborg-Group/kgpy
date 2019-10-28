import dataclasses

__all__ = ['Material']


@dataclasses.dataclass
class Material:
    
    name: str = ''


mirror = Material(name='mirror')
