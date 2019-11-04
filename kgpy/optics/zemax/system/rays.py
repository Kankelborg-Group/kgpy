import typing as tp
import numpy as np
import nptyping as npt

from kgpy.optics.zemax import ZOSAPI


def trace(
        zemax_system: ZOSAPI.IOpticalSystem,

        input_rays_x: npt.Array[float],
        input_rays_y: npt.Array[float],
        configuration_indices: tp.Optional[tp.List[int]] = None,
        surface_indices: tp.List[int] = (~0,),
        wavelength_indices=None,

):

    input_rays = np.broadcast(input_rays_x, input_rays_y)

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




