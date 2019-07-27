
import typing as tp
import numpy as np
import astropy.units as u

from kgpy.optics.zemax import ZOSAPI


def raytrace(self,
             zemax_system: ZOSAPI.IOpticalSystem,
             configuration_index: int,
             surface_index: int,
             wavelength_indices: tp.List[int],
             field_x: u.Quantity,
             field_y: u.Quantity,
             pupil_x: u.Quantity,
             pupil_y: u.Quantity,
             ) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:


    # Store length of each axis in ray grid
    num_field_x = len(field_x)
    num_field_y = len(field_y)
    num_pupil_x = len(pupil_x)
    num_pupil_y = len(pupil_y)

    # Create grid of rays
    Fx, Fy, Px, Py = np.meshgrid(field_x, field_y, pupil_x, pupil_y, indexing='ij')

    # Store shape of grid
    sh = list(Fx.shape)

    sh = [len(wavelength_indices)] + sh

    # Allocate output arrays
    V = np.empty(sh)  # Vignetted rays
    X = np.empty(sh)
    Y = np.empty(sh)

    old_config = self.config

    zemax_system.MCE.SetCurrentConfiguration(configuration_index + 1)

    # Initialize raytrace
    rt = self.zos_sys.Tools.OpenBatchRayTrace()  # raytrace object
    tool = zemax_system.Tools.CurrentTool  # pointer to active tool

    surf = zemax_system.LDE.GetSurfaceAt(surface_index)


    # Open instance of batch raytrace
    rt_dat = rt.CreateNormUnpol(num_pupil_x * num_pupil_y, ZOSAPI.Tools.RayTrace.RaysType.Real,
                                surf.SurfaceNumber)

    # Run raytrace for each wavelength
    for w in wavelength_indices:

        wavl = zemax_system.SystemData.Wavelengths.GetWavelength(w + 1)

        # Run raytrace for each field angle
        for fi in range(num_field_x):
            for fj in range(num_field_y):

                rt_dat.ClearData()

                # Loop over pupil to add rays to batch raytrace
                for pi in range(num_pupil_x):
                    for pj in range(num_pupil_y):

                        # Select next ray
                        fx = Fx[fi, fj, pi, pj]
                        fy = Fy[fi, fj, pi, pj]
                        px = Px[fi, fj, pi, pj]
                        py = Py[fi, fj, pi, pj]

                        # Write ray to pipe
                        rt_dat.AddRay(wavl.WavelengthNumber, fx, fy, px, py,
                                      ZOSAPI.Tools.RayTrace.OPDMode.None_)

                # Execute the raytrace
                tool.RunAndWaitForCompletion()

                # Initialize the process of reading the results of the raytrace
                rt_dat.StartReadingResults()

                # Loop over pupil and read the results of the raytrace
                for pi in range(num_pupil_x):
                    for pj in range(num_pupil_y):
                        # Read next result from pipe
                        (ret, n, err, vig, x, y, z, l, m, n, l2, m2, n2, opd,
                         I) = rt_dat.ReadNextResult()

                        # Store next result in output arrays
                        V[w, fi, fj, pi, pj] = vig
                        X[w, fi, fj, pi, pj] = x
                        Y[w, fi, fj, pi, pj] = y

    tool.Close()

    self.config = old_config

    return V, X, Y
