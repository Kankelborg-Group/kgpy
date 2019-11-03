
from unittest import TestCase
import copy
import numpy as np
from scipy.optimize import curve_fit
import astropy
import astropy.units as u
import matplotlib.pyplot as plt
from ndcube import NDCube

class Line:

    def __init__(self, ion, intensity, center, width):

        self.ion = ion
        self.intensity = intensity
        self.center = center
        self.width = width

    def transform(self, cube, new_line, wavl_axis=-1):

        # find coordinates of old line center
        _, _, old_wavl_ind = cube.world_to_pixel(0 * u.deg, 0 * u.deg, self.center)

        # find ratio of widths to scale the pixels by
        width_ratio = float(new_line.width / self.width)


        # Make a copy of the coordinate system to modify
        new_wcs = cube.wcs.deepcopy()

        # Add modifications to coordinate system
        new_wcs.wcs.crval[0] = new_line.center.to(u.m).value    # set line center value
        new_wcs.wcs.crpix[0] = old_wavl_ind.value       # set line center pixel
        cdelt = new_wcs.wcs.cdelt   # set width of pixels
        cdelt[0] *= width_ratio
        new_wcs.wcs.cdelt = cdelt

        # construct a new cube from the new coordinate system
        # cube = copy.deepcopy(cube)
        new_cube = NDCube(cube.data, new_wcs)


        return new_cube


    def calc_ave_gaussian_fit(self, cube, avg_axis=None, wavl_axis=-1):

        # Check that the wavlength axis is a single integer
        if type(wavl_axis) != int:
          raise ValueError('Expected integer wavelength axis')

        # Make sure avg_axis is the type of input we expect
        if avg_axis is not None:

            # Convert to tuple if integer
            if (type(avg_axis) is int):
                avg_axis = (avg_axis,)

            # Check that we're not averaging along the wavelength axis
            if wavl_axis in avg_axis:
                raise ValueError('Wavelength axis may not be averaged along')

        # Extract ndarray from the NDcube object
        data = cube.data

        # Calculate absolute axis index
        wavl_axis = wavl_axis % len(data.shape)

        # Perform the mean along the specified axes
        if avg_axis is None:
            mean = data
        else:
            mean = np.mean(data, axis=avg_axis, keepdims=True)

        # Calculate new index of wavlength axis after mean
        # if avg_axis is not None:
        #     old_axes = np.arange(len(data.shape))
        #     new_axes = np.delete(old_axes, avg_axis)
        #     wavl_axis = np.where(new_axes == wavl_axis % len(data.shape))[0][0]


        # Find shape of result by removing wavlength axis from tuple
        res_shape = list(mean.shape)
        res_shape[wavl_axis] = 1

        # Retrieve values of wavelength axis bins
        l = cube.axis_world_coords(wavl_axis)

        # Allocate memory to store results
        height = np.empty(res_shape)
        center = np.empty(res_shape) * l.unit
        width = np.empty(res_shape) * l.unit
        offset = np.empty(res_shape)
        noise = np.empty(res_shape)

        guess = [self.intensity, float(self.center / l.unit), float(self.width / l.unit), 0]
        lower_bounds = [0.0, l[0].value, 0.0, -10]
        upper_bounds = [1e5, l[-1].value, 1e-10, 10]

        for res_ind, _ in np.ndenumerate(height):

            mean_ind = list(res_ind)
            mean_ind[wavl_axis] = Ellipsis
            mean_ind = tuple(mean_ind)


            # plt.plot(l, mean[mean_ind])
            # plt.plot(l, self.gauss(l.value, guess[0], guess[1], guess[2], guess[3]))
            if np.sum(mean[mean_ind]) == 0.0:
                height[res_ind] = np.nan
                center[res_ind] = np.nan
                width[res_ind] = np.nan
                offset[res_ind] = np.nan
                noise[res_ind] = np.nan

            else:
                # p, _ = curve_fit(self.gauss, l.value, mean[mean_ind], p0=guess, bounds=(lower_bounds, upper_bounds))
                p, _ = curve_fit(gauss, l.value, mean[mean_ind], p0=guess)

                height[res_ind] = p[0]
                center[res_ind] = p[1] * l.unit
                width[res_ind] = p[2] * l.unit
                offset[res_ind] = p[3]

                y = gauss(l.value, p[0], p[1], p[2], p[3])

                noise[res_ind] = np.std(mean[mean_ind] - y)

            #
            # plt.plot(l,y)
            # plt.show()

        return height.squeeze(), center.squeeze(), width.squeeze(), offset.squeeze(), noise.squeeze()

def gauss(x, a, x0, w, b):

    X = x - x0

    return a * np.exp(- X * X / (w * w)) + b


class TestLine(TestCase):

    def setUp(self):

        self.Si_name = 'Si IV'
        self.Si_I0 = 10
        self.Si_l0 = 1394 * u.AA
        self.Si_dl = .50 * u.AA

        self.Si_line = Line(ion=self.Si_name, intensity=self.Si_I0, center=self.Si_l0, width=self.Si_dl)

        # wavelength limits
        self.l_min = 1380 * u.AA
        self.l_max = 1410 * u.AA

        # space-x limits
        self.x_min = 0 * u.arcsec
        self.x_max = 60 * u.arcsec

        # space y-limits
        self.y_min = 0 * u.arcsec
        self.y_max = 30 * u.arcsec

        nl = 255
        ny = 255
        nx = 255

        self.cube = self.gen_cube(nx, ny, nl, self.Si_I0, self.Si_l0, self.Si_dl)


    def tearDown(self):
        plt.show()

    def gen_cube(self, nx, ny, nl, intensity, center, width):

        dl = (self.l_max - self.l_min) / float(nl - 1)
        dy = (self.y_max - self.y_min) / float(ny - 1)
        dx = (self.x_max - self.x_min) / float(nx - 1)

        # WCS test dictionary
        wcs_input_dict = {'CTYPE1': 'WAVE    ', 'CUNIT1': str(dl.unit), 'CDELT1': dl.value, 'CRPIX1': 0,
                          'CRVAL1': self.l_min.value, 'NAXIS1': nl,
                          'CTYPE2': 'HPLT-TAN', 'CUNIT2': str(dy.unit), 'CDELT2': dy.value, 'CRPIX2': 0,
                          'CRVAL2': self.y_min.value, 'NAXIS2': ny,
                          'CTYPE3': 'HPLN-TAN', 'CUNIT3': str(dx.unit), 'CDELT3': dx.value, 'CRPIX3': 0,
                          'CRVAL3': self.x_min.value, 'NAXIS3': nx}

        # WCS test object
        input_wcs = astropy.wcs.WCS(wcs_input_dict)


        # Create storage for data array
        data = np.empty((nx, ny, nl))

        # Create test NDCube object
        cube = NDCube(data, input_wcs)

        _, _, l = cube.axis_world_coords()
        L = np.tile(l, (nx, ny, 1)) - center
        cube.data[:] = intensity * np.exp(-L * L / (width * width))


        return cube

    def test_transform(self):

        He_l0 = 304 * u.AA
        He_dl = 0.1 *u.AA

        He_line = Line(center=He_l0, width=He_dl)

        self.cube.plot()

        trans_cube = self.Si_line.transform(self.cube, He_line)

        trans_cube.plot()

    def test_calc_ave_gaussian_fit(self):
        # self.Si_line.calc_ave_guassian_fit(self.cube)
        # height, center, width, noise = self.Si_line.calc_ave_guassian_fit(self.cube, avg_axis=1)
        height, center, width, noise = self.Si_line.calc_ave_gaussian_fit(self.cube, avg_axis=(0,1))

        unit = self.Si_l0.unit


        self.assertAlmostEqual(height.flatten()[0], self.Si_I0, places=5)
        self.assertAlmostEqual(float(center.flatten()[0] / unit), float(self.Si_l0 / unit))
        self.assertAlmostEqual(float(width.flatten()[0] / unit), float(self.Si_dl / unit))
        self.assertAlmostEqual(noise.flatten()[0], 0.0)
