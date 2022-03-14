import typing as typ
import numpy as np
import astropy.units as u
import dataclasses
import matplotlib.pyplot as plt
import time
from . import Result, antialias, forward


__all__ = ['SimpleMART']



@dataclasses.dataclass
class SimpleMART:
    type_int: typ.ClassVar[int] = 0
    track_cube_history: bool = False
    anti_aliasing: str = None
    rotation_kwargs: typ.Dict[str, typ.Any] = dataclasses.field(default_factory=lambda: {})
    verbose: bool = False

    """ `SimpleMART` is the basic unit of mart, and encompasses a single "filtering iteration".
    
    :param track_cube_history: if 'multiplicative', a copy of the cube after each multiplicative iteration will be stored. Very memory intensive.
                               if 'filter', a copy is saved each time MART converges (or exceeds max multiplicative iterations) 
    """

    # if rotation_kwargs is None:
    #     rotation_kwargs = {'reshape': False, 'prefilter': False, 'order': 3, 'mode': 'nearest', }

    @staticmethod
    def chisq(goodness_of_fit: np.ndarray) -> float:
        return np.nanmean(goodness_of_fit[goodness_of_fit != 0])


    @staticmethod
    def channel_is_not_converged(goodness_of_fit: np.ndarray) -> bool:
        chisq = SimpleMART.chisq(goodness_of_fit)
        return chisq > 1
        # return np.percentile(goodness_of_fit,99.9) > 1


    @staticmethod
    def correction_exponent(goodness_of_fit: np.ndarray) -> np.ndarray:
        return np.array(1)

    def __call__(
            self,
            results: Result,
            projections: 'np.ndarray[float]',
            projections_azimuth: u.Quantity,
            spectral_order: 'np.ndarray[int]',
            photon_read_noise: float = 1,
            max_multiplicative_iteration: int = 40,
            cube_shape: typ.Tuple[int, ...] = None,
            projections_offset_x: 'np.ndarray[int]' = np.array(0),
            projections_offset_y: 'np.ndarray[int]' = np.array(0),
            cube_offset_x: 'np.ndarray[int]' = np.array(0),
            cube_offset_y: 'np.ndarray[int]' = np.array(0),
            m_axis: int = ~4,
            a_axis: int = ~3,
            x_axis: int = ~2,
            y_axis: int = ~1,
            w_axis: int = ~0
    ) -> typ.NoReturn:
        """
        Multiplicative Algebraic Reconstruction Technique (MART) is an iterative reconstruction algorithm developed by
        Charles Kankelborg et al. at Montana State University
        :param projections: Real or synthetic ESIS observation. Must be at least 5-dimensional: spectral order, dispersion
        azimuth, spatial-x, spatial-y, wavelength. Note that the wavelength axis must be a singleton dimension. For ESIS, the
        spectral order axis will be a singleton dimension.
        :param projections_azimuth: 1D array of angles describing the dispersion direction. Must be the same length as
        `observation.shape[a_axis]`.
        :param spectral_order: 1D array of integers describing the spectral order of the dispersion. Must be same length as
        `observation.shape[m_axis]`.
        :param cube_shape: shape of the output cube. It is an error to specify both `cube_shape` and `cube_guess`.
        :param projections_offset_x:
        :param projections_offset_y:
        :param cube_offset_x:
        :param cube_offset_y:
        :param m_axis: in the data-cube, the index of the axis long which spectral order varies
        :param a_axis: in the data-cube, the index of the axis long which azimuthal projection angle varies
        :param x_axis: in the data-cube, the index of the axis long which x-position varies
        :param y_axis: in the data-cube, the index of the axis long which y-position varies
        :param w_axis: in the data-cube, the index of the axis long which wavelength varies
        :return: `Result` object containing the results.
        """

        n_channels = projections.shape[m_axis] * projections.shape[a_axis]
        r = results



        projections_azimuth, spectral_order = np.broadcast_arrays(projections_azimuth, spectral_order, subok=True)

        if self.verbose:
            print('Starting MART Iterations')
        for multiplicative_iter in range(max_multiplicative_iteration):
            n_converged = n_channels
            corrections = np.ones_like(r.cube)
            for m in range(projections.shape[m_axis]):
                for a in range(projections.shape[a_axis]):



                    sl = [slice(None)] * projections.ndim
                    sl[m_axis] = slice(m, m + 1)
                    sl[a_axis] = slice(a, a + 1)
                    projection = projections[tuple(sl)]  # type: np.ndarray[float]

                    test_projection = forward.model(
                        cube=r.cube,
                        projection_azimuth=projections_azimuth[a],
                        spectral_order=spectral_order[m],
                        projection_shape=projection.shape,
                        projection_spatial_offset=(projections_offset_x[m, a], projections_offset_y[m, a]),
                        cube_spatial_offset=(cube_offset_x[m, a], cube_offset_y[m, a]),
                        x_axis=x_axis,
                        y_axis=y_axis,
                        w_axis=w_axis,
                        rotation_kwargs=self.rotation_kwargs
                    )
                    test_projection[test_projection <= 0] = 0

                    if self.anti_aliasing == 'post':
                        test_projection = antialias.apply(test_projection, x_axis_index=x_axis, y_axis_index=y_axis)

                    goodness_of_fit = np.square(test_projection - projection) / (np.square(photon_read_noise) + test_projection)
                    chisq = SimpleMART.chisq(goodness_of_fit)

                    ### Useful if MART is going not converging.  Allows direct inspection of residuals.
                    # print(chisq)
                    # fig, ax = plt.subplots()
                    # im = ax.imshow(goodness_of_fit[0,0,:,:,0])
                    # fig.colorbar(im)
                    # plt.show()

                    # print(chisq,np.max(goodness_of_fit),goodness_of_fit[goodness_of_fit>1].size)

                    r.chisq_history.append(chisq)
                    r.mart_type_history.append(self.type_int)

                    if self.channel_is_not_converged(goodness_of_fit):
                        # print('Calculating Correction')
                        # this if-statement runs if if the mean of `goodness_of_fit` is greater than 1
                        n_converged -= 1
                        exponent = self.correction_exponent(goodness_of_fit)

                        # Adding this logic to protect against divide by zero errors
                        ratio = projection / test_projection
                        zeroes_loc = (test_projection == 0)
                        correction = ratio ** (2 * exponent / n_channels)
                        correction[zeroes_loc] = 1

                        deprojection = forward.deproject(
                            projection=correction,
                            projection_azimuth=projections_azimuth[a],
                            spectral_order=spectral_order[m],
                            cube_shape=cube_shape,
                            projection_spatial_offset=(projections_offset_x[m, a], projections_offset_y[m, a]),
                            cube_spatial_offset=(cube_offset_x[m, a], cube_offset_y[m, a]),
                            x_axis=x_axis,
                            y_axis=y_axis,
                            w_axis=w_axis,
                            rotation_kwargs=self.rotation_kwargs
                        )
                        deprojection[deprojection <= 0] = 0

                        # r.cube *= deprojection
                        corrections = corrections * deprojection

                        if self.track_cube_history == 'multiplicative':
                            r.cube_history.append(r.cube.copy())

            if n_converged == n_channels:
                if self.verbose:
                    print('MART Converged at iteration ', multiplicative_iter)
                if self.track_cube_history == 'filter':
                    r.cube_history.append(r.cube.copy())
                break

            # print('Correcting Guess Cube')
            r.cube *= corrections ** (1 / (n_channels - n_converged))

        if n_converged != n_channels:

            if self.track_cube_history == 'filter':
                r.cube_history.append(r.cube.copy())
            # warnings.warn('MART failed to converge, maximum number of iterations exceeded!')
            print('MART failed to converge, maximum number of iterations exceeded!')
