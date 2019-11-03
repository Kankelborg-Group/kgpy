
import numpy as np
import math
import astropy.units as u
import astropy.constants as const

from kso.model import Model, TestModel


class Passband(Model):

    pass


class RectangularPassband(Passband):

    def __init__(self, wavl_min, wavl_max):

        self.l_min = wavl_min
        self.l_max = wavl_max

    def __call__(self, cube):

        # Convert wavelength range to pixel indices
        _, _, k_min = cube.world_to_pixel(0 * u.deg, 0 * u.deg, self.l_min)
        _, _, k_max = cube.world_to_pixel(0 * u.deg, 0 * u.deg, self.l_max)

        _,_, l1 = cube.pixel_to_world(0 * u.pix, 0 *u.pix, 1*u.pix)
        _, _, l0 = cube.pixel_to_world(0 * u.pix, 0 * u.pix, 0 * u.pix)



        print('k min/max', k_min.value, k_max.value)

        # convert pixel Quantities to ints
        k_min = math.ceil(k_min.value)
        k_max = math.ceil(k_max.value)

        print('k min/max', k_min, k_max)

        # Crop cube
        cube = cube[:, :, k_min:k_max+1]

        return cube

class RectangularVelocityPassband(RectangularPassband):

    def doppler(self, wavl, vel):
        print('velocity_ratio', float(vel / const.c))
        return wavl / (1.0 + float(vel / const.c))

    def __init__(self, wavl, v_min, v_max):

        l_min = self.doppler(wavl, v_max)
        l_max = self.doppler(wavl, v_min)

        print(l_min, l_max)

        super().__init__(l_min, l_max)

class TestRectangularPassband(TestModel):

    def setUp(self):

        super().setUp()

    def tearDown(self):

        super().tearDown()

    def test__call__(self):

        # Establish limits of rectangular passband
        new_l_min = self.l_min + 2 * u.AA
        new_l_max = self.l_max - 2 * u.AA

        # Create rectangular passband model
        pb = RectangularPassband(new_l_min, new_l_max)

        for cube in self.cubes:

            cube = pb(cube)

            cube.plot()



