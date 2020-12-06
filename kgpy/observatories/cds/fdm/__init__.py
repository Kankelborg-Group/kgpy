import astropy.time
from kgpy import obs

__all__ = ['index', 'Cube']


index = [
    ['1996-04-24T07:01:15', '1996-04-24T15:37:59'],
    ['1996-05-12T02:55:11', '1996-05-12T11:31:03'],
    ['1996-06-12T17:04:19', '1996-06-13T01:46:26'],
]


class Cube(obs.spectral.Cube):
    pass
