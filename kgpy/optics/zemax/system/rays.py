import typing as tp
import numpy as np
import nptyping as npt
import astropy.units as u
import numba

from kgpy.optics.zemax import ZOSAPI

__all__ = ['trace']


def trace(
        zemax_system: ZOSAPI.IOpticalSystem,
        zemax_units: u.Unit,
        num_pupil: tp.Union[int, tp.Tuple[int, int]] = 5,
        num_field: tp.Union[int, tp.Tuple[int, int]] = 5,
        mask: npt.Array[bool] = None,
        configuration_indices: tp.Optional[tp.List[int]] = None,
        surface_indices: tp.List[int] = (~0,),
        wavelength_indices=None,
) -> tp.Tuple[u.Quantity, u.Quantity, np.ndarray]:

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

    surface_indices = np.array(surface_indices) % zemax_system.LDE.NumberOfSurfaces
    print(surface_indices)

    if wavelength_indices is None:
        wavelength_indices = np.arange(zemax_system.SystemData.Wavelengths.NumberOfWavelengths)

    starting_config = zemax_system.MCE.CurrentConfiguration

    total_num_rays = 100 * 100

    sh = (
        len(configuration_indices),
        len(surface_indices),
        len(wavelength_indices),
    )

    sh += fg_x.shape

    out_x = np.empty(sh)
    out_y = np.empty(sh)
    out_mask = np.empty(sh, dtype=int)

    for c, config_index in enumerate(configuration_indices):

        zemax_system.MCE.SetCurrentConfiguration(config_index + 1)

        rt = zemax_system.Tools.OpenBatchRayTrace()
        tool = zemax_system.Tools.CurrentTool

        for s, surf_index in enumerate(surface_indices):

            rt_dat = rt.CreateNormUnpol(total_num_rays, ZOSAPI.Tools.RayTrace.RaysType.Real, surf_index + 2)

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
                                    out_mask[c, s, w, ix, iy, jx, jy] = int(not(err == 0 and vig == 0))
                                    out_x[c, s, w, ix, iy, jx, jy] = x
                                    out_y[c, s, w, ix, iy, jx, jy] = y

                            rt_dat.ClearData()

        tool.Close()

    zemax_system.MCE.SetCurrentConfiguration(starting_config)

    out_x = out_x * zemax_units
    out_y = out_y * zemax_units

    return out_x, out_y, out_mask
