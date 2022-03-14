import dataclasses
import typing as typ
import numpy as np
import astropy.units as u
import scipy

from . import antialias, Result, SimpleMART, LGOFMART, forward


# Looking for Charles' original MART? it is located at:
# esis.data.inversion.moses_mart.moses

@dataclasses.dataclass
class MART:
    use_maximize: bool = True
    use_filter: bool = True
    anti_aliasing: str = None
    use_lgof: bool = True
    contrast_exponent: float = 0.2
    max_filtering_iterations: int = 10
    max_multiplicative_iteration: int = 40
    photon_read_noise: float = 1
    simple_mart: SimpleMART = None
    lgof_mart: LGOFMART = None
    track_cube_history: bool = False
    rotation_kwargs: typ.Dict[str, typ.Any] = dataclasses.field(default_factory=lambda: {}, )
    verbose: bool = False

    """ MART is the Multiplicative Algebraic Reconstruction Technique, developed here for use and application in 
    general slitless imaging spectrograph, such as the Multi-Order Solar EUV Spectrograph (MOSES) and the EUV Snapshot
    Snapshot Imaging Spectrograph (ESIS) instruments constructed and launched by the Kankelborg Group at Montana State
    University, Bozeman. MART was originally developed to reconstruct images in MOSES, and has since been generalized.
    
    MART is built as a callable object. Parameters determined during construction of a MART object are pertinent to how
    the algorithm will be carried out. When the MART object is called, input data is then specified.
    
    :param use_maximize: if True, use the `maximize` method during each filtering iteration.
    :param use_filter: if True, apply the `filter` method during each filtering iteration.
    :param anti_aliasing: 'before' means that the test projection is anti-aliased before the goodness of fit is
        determined in each multiplicative iteration. 'after' means the de-projection is anti-aliased. 'single' means 
        that the indivudual projections are anti-aliased only once, before beginning the multiplicative or filtering 
        iterations.
    :param use_lgof: if True, use the Local Goodness of Fit routine.
    :param contrast_exponent: During contrast enhancement, what scalar value is used as the exponent on the data.
    :param max_filtering_iterations: maximum number of filtering iterations to do
    :param max_multiplicative: maximum number of multiplicative iterations to do within a single filtering iteration.
    :param simple_mart:
    :param lgof_mart:
    :param track_cube_history: if True, a copy of the cube after each iteration will be stored. Very memory intensive,
        and not currently working.
    """

    def __post_init__(self):
        # if self.rotation_kwargs is None:
        #     self.rotation_kwargs = {'reshape': False, 'prefilter': False, 'order': 3, 'mode': 'nearest', }

        if self.simple_mart is None:
            self.simple_mart = SimpleMART(track_cube_history=self.track_cube_history,
                                          anti_aliasing=self.anti_aliasing,
                                          rotation_kwargs=self.rotation_kwargs,
                                          verbose=self.verbose)

        if self.lgof_mart is None:
            self.lgof_mart = LGOFMART(track_cube_history=self.track_cube_history,
                                      anti_aliasing=self.anti_aliasing,
                                      rotation_kwargs=self.rotation_kwargs,
                                      verbose=self.verbose)

    @staticmethod
    def maximize(
            cube: 'np.ndarray[float]',
    ) -> float:
        """
        Maximize function for use in MART, developed from CCK's `entropy` and `negentropy`
        :param cube:
        :return:
        """
        signal = cube.copy().flatten()
        # signal = signal[signal !=0] # attempting to account for zero paddings influence on entropy
        signal *= 100  # adding resolution to negentropy ... so noticeable difference in behavior
        signal = np.round(signal)
        unique_values, unique_counts = np.unique(signal, return_counts=True)
        probability = unique_counts / signal.size
        # print(probability)
        entropy = np.nansum(probability * np.log2(1 / probability))
        return -entropy

    @staticmethod
    def generate_filtering_kernel(
            dimensions: int,
            x_axis: int = ~2,
            y_axis: int = ~1,
            w_axis: int = ~0
    ) -> np.ndarray:
        """
        Generate the kernel used during the filtering iterations. Takes standard 1-D kernel of [0.25, 0.5, 0.25] and
        generalizes it to a given number of dimensions.
        :param dimensions: number of dimensions the kernel needs to be
        :param x_axis: index of the x-axis
        :param y_axis: index of the y-axis
        :param w_axis: index of the wavelength axis
        :return:
        """
        # This kernel is handed down from Charles
        kernel = np.array([0.25, 0.5, 0.25])

        ksh = [1] * dimensions
        ksh[w_axis] = kernel.shape[0]
        kernel = kernel.reshape(ksh)
        kernel_w = kernel
        kernel_x = np.moveaxis(kernel.copy(), w_axis, x_axis)
        kernel_y = np.moveaxis(kernel.copy(), w_axis, y_axis)
        kernel = kernel_x * kernel_y * kernel_w
        kernel /= kernel.sum()

        return kernel

    def filter(
            self,
            cube: 'np.ndarray[float]',
            kernel: 'np.ndarray[float]',
    ) -> 'np.ndarray[float]':
        """
        Filter for use in MART, developed from CCK's `contrast_smooth` function.
        :param cube: input array
        :param kernel: the kernel to use for the convolution
        :return: filtered version of input array.
        """

        # cube_exp = cube ** self.contrast_exponent
        cube_exp = (cube / 1) ** self.contrast_exponent

        cube_filtered = cube * (1 + cube_exp)

        # cube_filtered = astropy.convolution.convolve(cube_filtered, kernel, boundary='extend')
        cube_filtered = scipy.ndimage.convolve(cube_filtered, kernel, mode='constant', cval=0)
        cube_filtered *= cube.sum() / cube_filtered.sum()

        return cube_filtered

    # @profile(precision=2)
    def __call__(
            self,
            projections: 'np.ndarray[float]',
            projections_azimuth: u.Quantity = None,
            spectral_order: 'np.ndarray[int]' = None,
            cube_shape: typ.Tuple[int, ...] = None,
            cube_guess: 'typ.Optional[np.ndarray[float]]' = None,
            projections_offset_x: 'np.ndarray[int]' = np.array(0),
            projections_offset_y: 'np.ndarray[int]' = np.array(0),
            cube_offset_x: 'np.ndarray[int]' = np.array(0),
            cube_offset_y: 'np.ndarray[int]' = np.array(0),
            m_axis: int = ~4,
            a_axis: int = ~3,
            x_axis: int = ~2,
            y_axis: int = ~1,
            w_axis: int = ~0
    ) -> Result:
        """
        Multiplicative Algebraic Reconstruction Technique (MART) is an iterative reconstruction algorithm developed by
        Charles Kankelborg et al. at Montana State University
        :param projections: Real or synthetic ESIS observation. Must be at least 5-dimensional: spectral order, dispersion
        azimuth, spatial-x, spatial-y, wavelength. Note that the wavelength axis must be a singleton dimension. For ESIS, the
        spectral order axis will be a singleton dimension.
        :param projections_azimuth: 1D array of angles describing the dispersion direction. Must be the same length as
        `observation.shape[a_axis]`.
        :param spectral_order: 1D array of integers describing the spectral order of the dispersion. Must be same length as
        `observation.shape[m_axis]`. Defaults to 1.
        :param cube_shape: shape of the output cube. It is an error to specify both `cube_shape` and `cube_guess`.
        :param cube_guess: This the initial cube_guess at the inverted observation (the reconstruction). Must be 5-dimensional:
        spectral order, dispersion azimuth, spatial-x, spatial-y, wavelength.  Note that the spectral order axis and
        dispersion axis are singleton dimensions. It is an error to specify both `cube_shape` and `cube_guess`.
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
        if spectral_order is None:
            spectral_order = np.array([1])

        if projections_azimuth is None:
            projections_azimuth = [0] * u.deg

        n_channels = projections.shape[m_axis] * projections.shape[a_axis]
        r = Result()

        # assign "metadata" parameters to the Results object
        r.call_parameters = {
            'projections': projections,
            'projections_azimuth': projections_azimuth,
            'spectral_order': spectral_order,
            'cube_shape': cube_shape,
            'cube_guess': cube_guess,
            'projections_offset_x': projections_offset_x,
            'projections_offset_y': projections_offset_y,
            'cube_offset_x': cube_offset_x,
            'cube_offset_y': cube_offset_y,
            'm_axis': m_axis,
            'a_axis': a_axis,
            'x_axis': x_axis,
            'y_axis': y_axis,
            'w_axis': w_axis,
        }
        r.object_parameters = {
            'use_maximize': self.use_maximize,
            'use_filter': self.use_filter,
            'anti_aliasing': self.anti_aliasing,
            'use_lgof': self.use_lgof,
            'contrast_exponent': self.contrast_exponent,
            'photon_read_noise': self.photon_read_noise,
            'max_filtering_iterations': self.max_filtering_iterations,
            'max_multiplicative_iterations': self.max_multiplicative_iteration,
            'simple_mart': self.simple_mart,
            'lgof_mart': self.lgof_mart,
            'track_cue_history': self.track_cube_history,
            'rotation_kwargs': self.rotation_kwargs,
            'verbose': self.verbose
        }

        projection_shape = (projections.shape[m_axis], projections.shape[a_axis])

        projections_offset_x = np.broadcast_to(projections_offset_x, projection_shape)
        projections_offset_y = np.broadcast_to(projections_offset_y, projection_shape)

        cube_offset_x = np.broadcast_to(cube_offset_x, projection_shape)
        cube_offset_y = np.broadcast_to(cube_offset_y, projection_shape)

        if cube_shape is None:
            if cube_guess is None:
                raise ValueError('At least one of `cube_shape` and `cube_guess` must be specified.')
            else:
                cube_shape = cube_guess.shape
                cube_guess = cube_guess.copy()
        else:
            if cube_guess is not None:
                raise ValueError('It is an error to specify both `cube_shape` and `cube_guess`')
            else:
                cube_guess = np.ones(cube_shape)
                cube_guess /= n_channels * cube_guess.sum()

        r.cube = cube_guess

        projections = projections.copy()

        if self.anti_aliasing == 'pre':
            projections = antialias.apply(projections, x_axis_index=x_axis, y_axis_index=y_axis)

        filtering_kernel = self.generate_filtering_kernel(
            dimensions=r.cube.ndim,
            x_axis=x_axis,
            y_axis=y_axis,
            w_axis=w_axis
        )

        for filtering_iter in range(max(self.max_filtering_iterations, 1)):

            if self.verbose:
                print('---------------------------------------------')
                print('Filtering Iteration Number ', filtering_iter)

            if self.use_filter:
                r.cube = self.filter(r.cube, kernel=filtering_kernel)

            self.simple_mart(r, projections, projections_azimuth, spectral_order,
                             self.photon_read_noise, self.max_multiplicative_iteration,
                             cube_shape, projections_offset_x, projections_offset_y, cube_offset_x, cube_offset_y,
                             m_axis, a_axis, x_axis, y_axis, w_axis, )

            if self.use_lgof:
                self.lgof_mart(r, projections, projections_azimuth, spectral_order,
                               self.photon_read_noise, self.max_multiplicative_iteration,
                               cube_shape, projections_offset_x, projections_offset_y, cube_offset_x, cube_offset_y,
                               m_axis, a_axis, x_axis, y_axis, w_axis, )

            if self.use_maximize:
                norm = self.maximize(np.sqrt(r.cube))

                if filtering_iter > 0:
                    if norm > np.max(r.norm_history):
                        r.best_cube = r.cube
                        r.best_filtering_iteration = filtering_iter
                else:
                    r.best_cube = r.cube

                r.norm_history.append(norm)

        if not self.use_maximize:
            r.best_cube = r.cube
        return r
