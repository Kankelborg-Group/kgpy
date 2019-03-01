
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from ndcube import NDCube

from kso.model import Model, TestModel
from kso.ImagingSpectograph.ComputedTomography import ZeroOrderDispersion
from kso.ndcube import rebin

class Binning(Model):

    def __init__(self, plate_scale_x, plate_scale_y):

        self.plate_scale_x = plate_scale_x
        self.plate_scale_y = plate_scale_y

    def __call__(self, cube):

        # Extract data array from cube
        data = cube.data

        _, x1, y1 = cube.pixel_to_world(1 * u.pix, 1 * u.pix, 1 * u.pix)
        _, x0, y0 = cube.pixel_to_world(0 * u.pix, 0 * u.pix, 0 * u.pix)

        old_ps_x = (x1 - x0) / u.pix
        old_ps_y = (y1 - y0) / u.pix


        ps_ratio_x = round(float(self.plate_scale_x / old_ps_x))
        ps_ratio_y = round(float(self.plate_scale_y / old_ps_y))



        print(ps_ratio_x, ps_ratio_y)

        rebin_factors = [1, ps_ratio_x, ps_ratio_y]
        new_cube = rebin(cube, rebin_factors)
        cube.plot()
        new_cube.plot()


        return new_cube

class TestBinning(TestModel):

    def setUp(self):

        super().setUp()

    def tearDown(self):

        super().tearDown()

    def test__call__(self):

        ps = 0.6 * u.arcsec / u.pix

        dm = ZeroOrderDispersion()
        bm = Binning(ps, ps)

        for cube in self.cubes:

            cube = dm(cube)
            bm(cube)


class PsuedoBinning(Binning):

    def __init__(self, plate_scale_x, plate_scale_y):

        super().__init__(plate_scale_x, plate_scale_y)


    def __call__(self, cube):

        # Extract data array from cube
        data = cube.data

        new_wcs = cube.wcs.deepcopy()
        new_wcs.wcs.cdelt[1] = self.plate_scale_y * u.pix / u.deg
        new_wcs.wcs.cdelt[2] = self.plate_scale_x * u.pix / u.deg

        new_cube = NDCube(data, new_wcs, uncertainty=cube.uncertainty, mask=cube.mask, meta=cube.meta, unit=cube.unit,
                          missing_axis=cube.missing_axis)

        cube.plot()
        new_cube.plot()



class TestPsuedoBinning(TestBinning):

    def setUp(self):

        super().setUp()

    def tearDown(self):

        super().tearDown()

    def test__call__(self):

        ps = 0.6 * u.arcsec / u.pix

        dm = ZeroOrderDispersion()
        bm = PsuedoBinning(ps, ps)

        for cube in self.cubes:

            cube = dm(cube)
            bm(cube)
