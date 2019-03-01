from abc import ABC, abstractmethod
import unittest
import numpy as np
from ndcube import NDCube
import astropy
import astropy.units as u
import matplotlib.pyplot as plt

class Model(ABC):

    @abstractmethod
    def __call__(self, cube):
        """
        Apply a forward model to cube
        :param cube: kso.cube object
        :return: new cube object
        """
        return cube
    #
    # @abstractmethod
    # def invert(self, cube):
    #     """
    #     Apply an inverse model to a cube
    #     :param cube:
    #     :return:
    #     """
    #     return cube

class IdentityModel(Model):
    def __call__(self, cube):
        return super().__call__(cube)

class TestModel(unittest.TestCase):

    def setUp(self):

        # wavelength limits
        self.l_min = 10 * u.AA
        self.l_max = 20 * u.AA

        # space-x limits
        self.x_min = 0 * u.arcsec
        self.x_max = 60 * u.arcsec

        # space y-limits
        self.y_min = 0 * u.arcsec
        self.y_max = 30 * u.arcsec

        nl = 63
        ny = 255
        nx = 255

        self.cube_odd = self.gen_cube(nx, ny, nl)

        nl = 64
        ny = 256
        nx = 256

        self.cube_even = self.gen_cube(nx, ny, nl)

        self.cubes = [self.cube_even, self.cube_odd]

    def tearDown(self):

        plt.show()


    def gen_cube(self, nx, ny, nl):

        dl = (self.l_max - self.l_min) / float(nl - 1)
        dy = (self.y_max - self.y_min) / float(ny - 1)
        dx = (self.x_max - self.x_min) / float(nx - 1)

        # WCS test dictionary
        wcs_input_dict = {'CTYPE1': 'WAVE    ', 'CUNIT1': str(dl.unit), 'CDELT1': dl.value, 'CRPIX1': 0, 'CRVAL1': self.l_min.value, 'NAXIS1': nl,
                          'CTYPE2': 'HPLT-TAN', 'CUNIT2': str(dy.unit), 'CDELT2': dy.value, 'CRPIX2': 0, 'CRVAL2': self.y_min.value, 'NAXIS2': ny,
                          'CTYPE3': 'HPLN-TAN', 'CUNIT3': str(dx.unit), 'CDELT3': dx.value, 'CRPIX3': 0, 'CRVAL3': self.x_min.value, 'NAXIS3': nx}

        # WCS test object
        input_wcs = astropy.wcs.WCS(wcs_input_dict)

        rl = (nl - 1) / 2
        ry = (ny - 1) / 2
        rx = (nx - 1) / 2

        l = np.arange(-rl, rl + 1)
        y = np.arange(-ry, ry + 1)
        x = np.arange(-rx, rx + 1)
        X, Y, L = np.meshgrid(x, y, l, indexing='ij')

        # Create random data array
        data = np.exp(-X * X / (rx * rx)) * np.exp(- Y * Y / (ry * ry)) * np.exp(-L * L / (rl * rl))

        # Create test NDCube object
        cube = NDCube(data, input_wcs)

        return cube

