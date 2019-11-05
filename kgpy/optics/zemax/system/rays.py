import typing as tp
import numpy as np
import nptyping as npt
import astropy.units as u

from kgpy.optics.zemax import ZOSAPI


def trace(
        zemax_system: ZOSAPI.IOpticalSystem,
        num_pupil: tp.Union[int, tp.Tuple[int, int]] = 5,
        num_field: tp.Union[int, tp.Tuple[int, int]] = 5,
        mask: npt.Array[bool] = None,
        configuration_indices: tp.Optional[tp.List[int]] = None,
        surface_indices: tp.List[int] = (~0,),
        wavelength_indices=None,

):
    if isinstance(num_field, int):
        num_field = num_field, num_field

    if isinstance(num_pupil, int):
        num_pupil = num_pupil, num_pupil

    field_x = np.linspace(-1, 1, num_field[0])
    field_y = np.linspace(-1, 1, num_field[1])
    pupil_x = np.linspace(-1, 1, num_pupil[0])
    pupil_y = np.linspace(-1, 1, num_pupil[1])

    fg_x, fg_y, pg_x, pg_y = np.meshgrid(field_x, field_y, pupil_x, pupil_y, indexing='ij')

    if configuration_indices is None:
        configuration_indices = np.arange(zemax_system.MCE.NumberOfConfigurations)

    surface_indices = np.array(surface_indices)
    if wavelength_indices is None:
        wavelength_indices = np.arange(zemax_system.SystemData.Wavelengths.NumberOfWavelengths)

    old_config = zemax_system.MCE.CurrentConfiguration

    total_num_rays = 100 * 100

    for c in configuration_indices:

        zemax_system.MCE.SetCurrentConfiguration(c + 1)

        rt = zemax_system.Tools.OpenBatchRayTrace()
        tool = zemax_system.Tools.CurrentTool

        for s in surface_indices:

            rt_dat = rt.CreateNormUnpol(total_num_rays, ZOSAPI.Tools.RayTrace.RaysType.Real, s + 1)

            for w in wavelength_indices:

                pass



    zemax_system.MCE.SetCurrentConfiguration(old_config)
