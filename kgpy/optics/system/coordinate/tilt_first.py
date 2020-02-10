import dataclasses

_all__ = ['TiltFirst', 'InverseTiltFirst']


@dataclasses.dataclass
class TiltFirst:
    value: bool = False
    
    def __invert__(self):
        return not self.value
    
    
@dataclasses.dataclass
class InverseTiltFirst:
    
    _tilt_first: TiltFirst
    
    @property
    def value(self):
        return not self._tilt_first.value
