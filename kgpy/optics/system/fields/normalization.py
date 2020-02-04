import abc
import typing as typ
import numpy as np
import astropy.units as u

__all__ = ['Normalization', 'Rectangular', 'Radial']


class Normalization(abc.ABC):

    @abc.abstractmethod
    def normalize(self, x: u.Quantity, y: u.Quantity) -> typ.Tuple[u.Quantity, u.Quantity]:

        pass


class Rectangular(Normalization):

    def normalize(self, x: u.Quantity, y: u.Quantity) -> typ.Tuple[u.Quantity, u.Quantity]:

        norm_x = np.max(np.abs(x), axis=~0, keepdims=True)
        norm_y = np.max(np.abs(y), axis=~0, keepdims=True)

        return x / norm_x, y / norm_y


class Radial(Normalization):

    def normalize(self, x: u.Quantity, y: u.Quantity) -> typ.Tuple[u.Quantity, u.Quantity]:

        norm = np.max(np.sqrt(np.square(x) + np.square(y)), axis=~0, keepdims=True)

        return x / norm, y / norm
