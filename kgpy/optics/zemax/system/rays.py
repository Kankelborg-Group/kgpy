import typing as tp
import numpy as np
import nptyping as npt
import astropy.units as u

from kgpy.optics.system import Rays
from kgpy.optics.zemax import ZOSAPI

__all__ = ['trace']


def trace(
        zemax_system: ZOSAPI.IOpticalSystem,
        zemax_units: u.Unit,
        num_pupil: tp.Union[int, tp.Tuple[int, int]] = 5,
        num_field: tp.Union[int, tp.Tuple[int, int]] = 5,
        mask: npt.Array[bool] = None,
        configuration_indices: tp.Optional[tp.List[int]] = None,
        surface_indices: tp.Optional[tp.List[int]] = (~0,),
        wavelength_indices=None,
) -> Rays:
    """
    General Zemax raytracing function.
    Trace a specified number of pupil/field points through an optical system.
    :param zemax_system: Pointer to Zemax optical system.
    :param zemax_units: Lens units of the Zemax optical system.
    :param num_pupil: Number of pupil positions to sample.
    If scalar, this is the number of points per axis.
    If a pair of numbers, the first value is the number of pupil positions along the x-axis and the second value is the
    number of pupil positions along the y-axis.
    :param num_field: Number of field positions to sample.
    This argument can be either a scalar or a pair of numbers, with the same format as `num_pupil`.
    :param mask:
    :param configuration_indices:
    :param surface_indices:
    :param wavelength_indices:
    :return:
    """

    if isinstance(num_field, int):
        num_field = num_field, num_field

    if isinstance(num_pupil, int):
        num_pupil = num_pupil, num_pupil

    field_x = np.linspace(-1, 1, num_field[0])
    field_y = np.linspace(-1, 1, num_field[1])
    pupil_x = np.linspace(-1, 1, num_pupil[0])
    pupil_y = np.linspace(-1, 1, num_pupil[1])

    fg_x, fg_y, pg_x, pg_y = np.meshgrid(field_x, field_y, pupil_x, pupil_y, indexing='ij')

    if mask is None:
        mask = np.broadcast_to(1, fg_x.shape)

    if configuration_indices is None:
        configuration_indices = np.arange(zemax_system.MCE.NumberOfConfigurations)

    if surface_indices is None:
        surface_indices = np.arange(zemax_system.LDE.NumberOfSurfaces)
    surface_indices = np.array(surface_indices) % zemax_system.LDE.NumberOfSurfaces

    if wavelength_indices is None:
        wavelength_indices = np.arange(zemax_system.SystemData.Wavelengths.NumberOfWavelengths)

    starting_config = zemax_system.MCE.CurrentConfiguration

    total_num_rays = 100 * 100

    sh = (
        len(configuration_indices),
        len(surface_indices),
        len(wavelength_indices),
    ) + fg_x.shape
    
    rays = Rays.empty(sh)

    for c, config_index in enumerate(configuration_indices):

        zemax_system.MCE.SetCurrentConfiguration(config_index + 1)

        rt = zemax_system.Tools.OpenBatchRayTrace()
        tool = zemax_system.Tools.CurrentTool

        for s, surf_index in enumerate(surface_indices):

            gmatrix = zemax_system.LDE.GetGlobalMatrix(surf_index + 1)
            _, r11, r12, r13, r21, r22, r23, r31, r32, r33, x0, y0, z0 = gmatrix

            rt_dat = rt.CreateNormUnpol(total_num_rays, ZOSAPI.Tools.RayTrace.RaysType.Real, surf_index + 1)

            for w, wavl_index in enumerate(wavelength_indices):

                for ix, fx in enumerate(field_x):
                    for iy, fy in enumerate(field_y):

                        for jx, px in enumerate(pupil_x):
                            for jy, py in enumerate(pupil_y):
                                if mask[ix, iy, jx, jy] == 1:

                                    rt_dat.AddRay(wavl_index + 1, fx, fy, px, py, ZOSAPI.Tools.RayTrace.OPDMode.None_)

                            tool.RunAndWaitForCompletion()
                            rt_dat.StartReadingResults()

                            for jy, py in enumerate(pg_y):
                                if mask[ix, iy, jx, jy] == 1:
                                    zemax_ray = rt_dat.ReadNextResult()
                                    ret, n, err, vig, x, y, z, l, m, n, l2, m2, n2, opd, intensity = zemax_ray

                                    x_global = r11 * x + r12 * y + r13 * z + x0
                                    y_global = r21 * x + r22 * y + r23 * z + y0
                                    z_global = r31 * x + r32 * y + r33 * z + z0

                                    ind = c, s, w, ix, iy, jx, jy
                                    rays.x[ind] = x_global * zemax_units
                                    rays.y[ind] = y_global * zemax_units
                                    rays.z[ind] = z_global * zemax_units
                                    rays.cos_ax[ind] = l
                                    rays.cos_ay[ind] = m
                                    rays.cos_az[ind] = n
                                    rays.cos_nx[ind] = l2
                                    rays.cos_ny[ind] = m2
                                    rays.cos_nz[ind] = n2
                                    rays.mask[ind] = vig == 0 and err == 0

                            rt_dat.ClearData()

        tool.Close()

    zemax_system.MCE.SetCurrentConfiguration(starting_config)

    return rays
